"""
train.py — ReMemorize full training script.

Two modes:
  --dev    Skip LLM loading; use synthetic placeholder data + HeuristicRewardScorer.
           Exercises the entire GRPO + EW-RLS pipeline in seconds on CPU.
  (default) Full training with a HuggingFace LLM (BF16 + Flash-Attention-2),
           NLIRewardScorer, and real MIMIC / MTS-Dialog data.

Example (dev mode, no GPU required):
    python train.py --dev --num_epochs 2 --group_size 4

Example (full run):
    python train.py \
        --model mistralai/Mistral-7B-v0.1 \
        --dataset mimic \
        --train_file data/mimic_train.jsonl \
        --eval_file  data/mimic_eval.jsonl  \
        --num_epochs 5 --group_size 8 \
        --output_dir runs/memorize_v1
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from dataset import (
    MIMICDataset,
    MTSDialogDataset,
    ReMemorizeDataset,
    SummarizationExample,
    build_dataloader,
)
from memory_manager import ReMemorizeMemoryManager
from reward_manager import RewardCalibrator, RewardWeights
from scorers import HeuristicRewardScorer, NLIRewardScorer
from trainer import ReMemorizeTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("memorize")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ReMemorize")

    # Mode
    p.add_argument("--dev", action="store_true",
                   help="Dev mode: synthetic data + heuristic scorer, no LLM.")

    # Model
    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                   help="HuggingFace model id or local path.")
    p.add_argument("--freeze_llm", action="store_true",
                   help="Freeze LLM weights; only train memory interface.")
    p.add_argument("--no_flash_attn", action="store_true",
                   help="Disable Flash-Attention-2 (fallback to standard attention).")

    # Memory
    p.add_argument("--d_key",    type=int, default=64)
    p.add_argument("--d_value",  type=int, default=64)
    p.add_argument("--d_memory", type=int, default=128)

    # Dataset
    p.add_argument("--dataset",    type=str, default="mimic",
                   choices=["mimic", "mts_dialog"],
                   help="Which dataset to use (real files or placeholders in --dev).")
    p.add_argument("--train_file", type=str, default=None,
                   help="Path to training JSONL (required when not in --dev mode).")
    p.add_argument("--eval_file",  type=str, default=None,
                   help="Path to eval JSONL (optional).")
    p.add_argument("--max_train",  type=int, default=None,
                   help="Truncate training set to N examples.")
    p.add_argument("--placeholder_n", type=int, default=200,
                   help="Number of synthetic examples in --dev mode.")

    # Training
    p.add_argument("--num_epochs",  type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Documents per gradient step (keep at 1 for GRPO).")
    p.add_argument("--group_size",  type=int,   default=4,
                   help="G candidate summaries per document (GRPO group).")
    p.add_argument("--lr",          type=float, default=1e-4,
                   help="Learning rate for memory interface parameters.")
    p.add_argument("--ppo_lr",      type=float, default=1e-5,
                   help="Learning rate for the gating policy (PPO/GRPO).")
    p.add_argument("--max_new_tokens",     type=int, default=256)
    p.add_argument("--max_source_length", type=int, default=3840,
                   help="Max source tokens fed to LLM. 3840+256=4096 = Mistral context window.")
    p.add_argument("--max_target_length", type=int, default=256,
                   help="Max target tokens for teacher-forcing.")
    p.add_argument("--lm_loss_weight", type=float, default=0.5,
                   help="Weight of supervised CE loss relative to GRPO reward.")
    p.add_argument("--warmup_steps",   type=int, default=100)
    p.add_argument("--grad_clip",      type=float, default=1.0)

    # Reward
    p.add_argument("--scorer", type=str, default="nli",
                   choices=["heuristic", "nli"],
                   help="Reward scorer. 'nli' requires transformers + sentence-transformers.")
    p.add_argument("--reward_alpha", type=float, default=1.0)
    p.add_argument("--reward_beta",  type=float, default=0.3)
    p.add_argument("--reward_gamma", type=float, default=0.1)
    p.add_argument("--reward_delta", type=float, default=0.8)

    # Output / logging
    p.add_argument("--output_dir",    type=str, default="runs/memorize")
    p.add_argument("--save_every",    type=int, default=500,
                   help="Save checkpoint every N global steps.")
    p.add_argument("--eval_every",    type=int, default=200,
                   help="Run evaluation every N global steps (0 = end of epoch only).")
    p.add_argument("--log_every",     type=int, default=10)
    p.add_argument("--wandb",         action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="ReMemorize")
    p.add_argument("--seed",          type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores."""
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = max(len(predictions), 1)
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for k in totals:
                totals[k] += scores[k].fmeasure
        return {k: v / n for k, v in totals.items()}
    except ImportError:
        def _f1(pred: str, ref: str) -> float:
            p_toks = set(pred.lower().split())
            r_toks = set(ref.lower().split())
            if not p_toks or not r_toks:
                return 0.0
            inter = len(p_toks & r_toks)
            prec = inter / len(p_toks)
            rec  = inter / len(r_toks)
            return 2 * prec * rec / (prec + rec + 1e-9)
        scores = [_f1(p, r) for p, r in zip(predictions, references)]
        mean = sum(scores) / max(len(scores), 1)
        return {"rouge1": mean, "rouge2": 0.0, "rougeL": mean}


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1 and BLEU-2 scores."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        bleu1_total, bleu2_total = 0.0, 0.0
        n = max(len(predictions), 1)
        for pred, ref in zip(predictions, references):
            ref_toks  = [ref.lower().split()]
            pred_toks = pred.lower().split()
            bleu1_total += sentence_bleu(ref_toks, pred_toks, weights=(1, 0, 0, 0),    smoothing_function=smoother)
            bleu2_total += sentence_bleu(ref_toks, pred_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        return {"bleu1": bleu1_total / n, "bleu2": bleu2_total / n}
    except ImportError:
        log.warning("nltk not installed; skipping BLEU.")
        return {"bleu1": 0.0, "bleu2": 0.0}


def compute_bertscore(predictions: List[str], references: List[str], device: str = "cpu") -> Dict[str, float]:
    """Compute BERTScore F1."""
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(predictions, references, lang="en", device=device, verbose=False)
        return {"bertscore_f1": float(F1.mean())}
    except ImportError:
        log.warning("bert_score not installed; skipping BERTScore.")
        return {"bertscore_f1": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Dev-mode generate_summary (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _dev_generate(source: str, rng, variation: int = 0) -> str:
    """
    Produce a lightweight synthetic summary for dev-mode GRPO.
    Picks a random subset of source tokens to mimic different candidates.
    """
    words = source.split()
    rng.shuffle(words)
    take = max(10, len(words) // (3 + variation % 4))
    return " ".join(words[:take]) + "."


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    backbone,             # ReMemorizeLLM | None in dev mode
    eval_dataset: ReMemorizeDataset,
    args: argparse.Namespace,
    step: int,
    wb_run=None,
    dev_rng=None,
    device: str = "cpu",
) -> Dict[str, float]:
    preds: List[str] = []
    refs:  List[str] = []

    for ex in eval_dataset:
        ref = ex.reference or ex.target
        if backbone is None:
            # dev mode: cheap synthetic prediction
            pred = _dev_generate(ex.source, dev_rng, variation=0)
        else:
            backbone.encode_and_update_memory(ex.source)
            pred = backbone.generate_summary(ex.source, max_new_tokens=args.max_new_tokens)
        preds.append(pred)
        refs.append(ref)

    scores: Dict[str, float] = {}
    scores.update(compute_rouge(preds, refs))
    scores.update(compute_bleu(preds, refs))
    scores.update(compute_bertscore(preds, refs, device=device))

    log.info(
        "eval step=%d  R1=%.4f  R2=%.4f  RL=%.4f  BLEU1=%.4f  BLEU2=%.4f  BERTScore-F1=%.4f",
        step,
        scores["rouge1"], scores["rouge2"], scores["rougeL"],
        scores["bleu1"],  scores["bleu2"],
        scores["bertscore_f1"],
    )
    if wb_run is not None:
        wb_run.log({"eval/" + k: v for k, v in scores.items()}, step=step)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB ──────────────────────────────────────────────────────────────
    wb_run = None
    if args.wandb:
        try:
            import wandb
            wb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"memorize_{int(time.time())}",
            )
        except ImportError:
            log.warning("wandb not installed; skipping W&B logging.")

    # ── Memory manager ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    mm = ReMemorizeMemoryManager(
        d_key=args.d_key,
        d_value=args.d_value,
        d_memory=args.d_memory,
        seed=args.seed,
        device=device,
    )

    # ── Reward calibrator + weights ─────────────────────────────────────────
    calibrator = RewardCalibrator(
        ema_alpha=0.1, clip_z=2.5, factual_floor=0.5, factual_floor_penalty=0.2
    )
    reward_weights = RewardWeights(
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        gamma=args.reward_gamma,
        delta=args.reward_delta,
    )

    # ── Reward scorer ───────────────────────────────────────────────────────
    if args.scorer == "nli" and not args.dev:
        log.info("Loading NLI reward scorer (DeBERTa-v3 + MiniLM)...")
        scorer = NLIRewardScorer(device=device)
    else:
        log.info("Using heuristic reward scorer.")
        scorer = HeuristicRewardScorer()

    # ── Trainer (policy + reward) ───────────────────────────────────────────
    trainer = ReMemorizeTrainer(
        memory_manager=mm,
        reward_calibrator=calibrator,
        reward_scorer=scorer,
        reward_weights=reward_weights,
        config=TrainerConfig(
            ppo_lr=args.ppo_lr,
            clip_eps=0.2,
            entropy_coef=0.005,
            group_size=args.group_size,
            max_kl=0.02,
            mse_reward_weight=0.1,
            lr_warmup_steps=args.warmup_steps,
        ),
        seed=args.seed,
    )

    # ── LLM backbone (skipped in dev mode) ─────────────────────────────────
    backbone = None
    backbone_optimizer = None
    backbone_scheduler = None

    if not args.dev:
        log.info("Loading LLM backbone: %s", args.model)
        from llm_backbone import ReMemorizeLLM
        backbone = ReMemorizeLLM(
            model_name_or_path=args.model,
            memory_manager=mm,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            dtype=torch.bfloat16,
            gradient_checkpointing=True,
            use_flash_attention=not args.no_flash_attn,
            freeze_llm=args.freeze_llm,
        )
        backbone_optimizer = AdamW(
            backbone.memory_interface_parameters(),
            lr=args.lr,
            weight_decay=1e-2,
        )
        if not args.freeze_llm:
            backbone_optimizer.add_param_group(
                {"params": backbone.llm.parameters(), "lr": args.lr * 0.1}
            )
        backbone_scheduler = LinearLR(
            backbone_optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=args.warmup_steps,
        )

    # ── Dataset ─────────────────────────────────────────────────────────────
    dev_rng = __import__("random").Random(args.seed)

    if args.dev:
        DatasetCls = MIMICDataset if args.dataset == "mimic" else MTSDialogDataset
        train_dataset = DatasetCls.placeholder(n=args.placeholder_n, seed=args.seed)
        eval_dataset: Optional[ReMemorizeDataset] = DatasetCls.placeholder(n=20, seed=args.seed + 1)
        log.info("Dev mode: %d synthetic train / 20 eval examples.", args.placeholder_n)
    else:
        if args.train_file is None:
            raise ValueError("--train_file is required when not in --dev mode.")
        if args.dataset == "mimic":
            train_dataset = MIMICDataset.from_jsonl(args.train_file, max_examples=args.max_train)
            eval_dataset = MIMICDataset.from_jsonl(args.eval_file) if args.eval_file else None
        else:
            train_dataset = MTSDialogDataset.from_jsonl(args.train_file, max_examples=args.max_train)
            eval_dataset = MTSDialogDataset.from_jsonl(args.eval_file) if args.eval_file else None
        log.info(
            "Loaded %d train examples%s.",
            len(train_dataset),
            f" / {len(eval_dataset)} eval" if eval_dataset else "",
        )

    dataloader = build_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )

    # ── Training ─────────────────────────────────────────────────────────────
    global_step = 0
    best_rouge1 = 0.0

    for epoch in range(1, args.num_epochs + 1):
        log.info("═══ Epoch %d / %d ═══", epoch, args.num_epochs)
        epoch_grpo_loss = 0.0
        epoch_lm_loss   = 0.0
        epoch_mse       = 0.0
        epoch_steps     = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}", unit="doc")
        for batch in pbar:
            for ex in batch:
                assert isinstance(ex, SummarizationExample)
                source    = ex.source
                reference = ex.reference or ex.target

                # ── 1. Encode source into memory / extract key_window ──────
                if backbone is not None:
                    key_window, _ = backbone.encode_and_update_memory(source)
                else:
                    # Dev mode: use random vectors as key proxies
                    T = min(len(source.split()), 64)
                    key_window = [
                        [dev_rng.gauss(0, 1) for _ in range(args.d_key)]
                        for _ in range(T)
                    ]
                    mm.reset_state()
                    mm.update(key_window, key_window)

                # ── 2. Generate G candidate summaries ──────────────────────
                if backbone is not None:
                    candidates = [
                        backbone.generate_summary(source, max_new_tokens=args.max_new_tokens)
                        for _ in range(args.group_size)
                    ]
                else:
                    candidates = [
                        _dev_generate(source, dev_rng, variation=g)
                        for g in range(args.group_size)
                    ]

                # ── 3. GRPO batch collection + policy update ────────────────
                grpo_batch = trainer.collect_grpo_batch(
                    key_windows=[key_window],
                    sources=[source],
                    summary_groups=[candidates],
                    references=[reference],
                )
                grpo_stats = trainer.grpo_step(grpo_batch)

                # ── 4. Supervised CE loss on gold target (backbone only) ────
                lm_loss_val = 0.0
                mse_val     = 0.0
                lm_info: Optional[dict] = None
                if backbone is not None and backbone_optimizer is not None:
                    backbone_optimizer.zero_grad()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=(device == "cuda")):
                        lm_loss, lm_info = backbone.forward_train(source, ex.target)
                        weighted_loss = args.lm_loss_weight * lm_loss
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        backbone.memory_interface_parameters(), args.grad_clip
                    )
                    backbone_optimizer.step()
                    if backbone_scheduler is not None:
                        backbone_scheduler.step()
                    lm_loss_val = lm_loss.item()
                    mse_val     = lm_info.get("mse", 0.0)

                # ── 5. Accumulate stats ────────────────────────────────────
                epoch_grpo_loss += grpo_stats.loss
                epoch_lm_loss   += lm_loss_val
                epoch_mse       += mse_val
                epoch_steps     += 1
                global_step     += 1

                # ── 6. Logging + safeguard monitoring ──────────────────────
                gamma_mean  = lm_info.get("gamma_mean",  0.0) if lm_info else 0.0
                gamma_std   = lm_info.get("gamma_std",   0.0) if lm_info else 0.0
                memory_norm = lm_info.get("memory_norm", 0.0) if lm_info else 0.0
                ce_loss_val = lm_info.get("ce_loss",     lm_loss_val) if lm_info else lm_loss_val

                # Safeguard 4: warn on collapse or explosion
                if backbone is not None and lm_info:
                    if gamma_mean > 0.8:
                        log.warning(
                            "step=%d: gamma_mean=%.3f > 0.8 — risk of memory overwriting. "
                            "Consider reducing ppo_lr or increasing detach_memory_hidden.",
                            global_step, gamma_mean,
                        )
                    if memory_norm > 50.0:
                        log.warning(
                            "step=%d: memory_norm=%.2f > 50 — potential explosion. "
                            "Check grad_clip and write_proj initialisation.",
                            global_step, memory_norm,
                        )

                pbar.set_postfix(
                    step=global_step,
                    grpo=f"{grpo_stats.loss:.4f}",
                    R1=f"{best_rouge1:.4f}",
                    γ=f"{gamma_mean:.3f}",
                )

                if global_step % args.log_every == 0:
                    log.info(
                        "step=%d  grpo=%.4f  ce=%.4f  lm=%.4f  "
                        "γ_mean=%.3f  γ_std=%.3f  mem_norm=%.2f  mse=%.4f",
                        global_step,
                        grpo_stats.loss,
                        ce_loss_val,
                        lm_loss_val,
                        gamma_mean,
                        gamma_std,
                        memory_norm,
                        mse_val,
                    )
                    if wb_run is not None:
                        wb_run.log(
                            {
                                "train/grpo_loss":   grpo_stats.loss,
                                "train/ce_loss":     ce_loss_val,
                                "train/lm_loss":     lm_loss_val,
                                "train/surrogate":   grpo_stats.surrogate,
                                "train/entropy":     grpo_stats.entropy,
                                "train/approx_kl":   grpo_stats.approx_kl,
                                "train/clip_frac":   grpo_stats.clip_fraction,
                                "train/gamma_mean":  gamma_mean,
                                "train/gamma_std":   gamma_std,
                                "train/memory_norm": memory_norm,
                                "train/mse":         mse_val,
                            },
                            step=global_step,
                        )

                # ── 7. Periodic evaluation ─────────────────────────────────
                if (
                    args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and eval_dataset is not None
                ):
                    scores = evaluate(backbone, eval_dataset, args, global_step, wb_run, dev_rng, device=device)
                    if scores["rouge1"] > best_rouge1:
                        best_rouge1 = scores["rouge1"]
                        if backbone is not None:
                            backbone.save_checkpoint(
                                str(output_dir / "best_ckpt"),
                                optimizer=backbone_optimizer,
                                scheduler=backbone_scheduler,
                                global_step=global_step,
                                best_rouge1=best_rouge1,
                            )
                            log.info("New best checkpoint saved (R1=%.4f).", best_rouge1)

                # ── 8. Periodic checkpoint ─────────────────────────────────
                if global_step % args.save_every == 0 and backbone is not None:
                    ckpt_dir = output_dir / f"step_{global_step}"
                    backbone.save_checkpoint(
                        str(ckpt_dir),
                        optimizer=backbone_optimizer,
                        scheduler=backbone_scheduler,
                        global_step=global_step,
                        best_rouge1=best_rouge1,
                    )
                    log.info("Checkpoint saved: %s", ckpt_dir)

        pbar.close()

        # ── End-of-epoch summary ───────────────────────────────────────────
        n = max(epoch_steps, 1)
        log.info(
            "Epoch %d done | avg_grpo_loss=%.4f  avg_lm_loss=%.4f  avg_mse=%.6f",
            epoch,
            epoch_grpo_loss / n,
            epoch_lm_loss / n,
            epoch_mse / n,
        )

        # End-of-epoch evaluation (if not already done mid-epoch)
        if eval_dataset is not None and (args.eval_every == 0 or epoch_steps < args.eval_every):
            scores = evaluate(backbone, eval_dataset, args, global_step, wb_run, dev_rng, device=device)
            if backbone is not None and scores["rouge1"] > best_rouge1:
                best_rouge1 = scores["rouge1"]
                backbone.save_checkpoint(
                    str(output_dir / "best_ckpt"),
                    optimizer=backbone_optimizer,
                    scheduler=backbone_scheduler,
                    global_step=global_step,
                    best_rouge1=best_rouge1,
                )

    # ── Final checkpoint ──────────────────────────────────────────────────
    if backbone is not None:
        backbone.save_checkpoint(
            str(output_dir / "final_ckpt"),
            optimizer=backbone_optimizer,
            scheduler=backbone_scheduler,
            global_step=global_step,
            best_rouge1=best_rouge1,
        )
        log.info("Final checkpoint saved.")

    if wb_run is not None:
        wb_run.finish()

    log.info("Training complete. Best ROUGE-1: %.4f", best_rouge1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
