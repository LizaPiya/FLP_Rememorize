#!/bin/bash
# wait_and_run.sh — Poll GPU and launch all pending jobs when memory is free.
# Usage: bash wait_and_run.sh
# Requires ~50 GiB free for training, ~15 GiB free for inference-only eval.

POLL_INTERVAL=120  # seconds between checks
REQUIRED_MiB=20000 # MiB needed for training runs (SOAP ~5GB, MIMIC ~16GB)

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/lizapiya/FLP_ReMemorizeProject

LOG=wait_and_run.log
# Shared lock file: prevents multiple script instances from launching GPU jobs
# simultaneously (race condition when two instances both see enough free memory).
GPU_LOCK=/tmp/rememorize_gpu.lock

echo "Watching GPU... will launch when ${REQUIRED_MiB} MiB free. Logging to $LOG"
echo "Press Ctrl+C to cancel."

# Helper: wait until GPU has enough free memory, then run a command.
# Uses flock on GPU_LOCK so only one job across all script instances runs at once.
# Retries indefinitely on OOM failure.
run_when_free() {
    local LABEL="$1"
    shift
    while true; do
        # Wait for enough free memory AND the GPU lock to be available
        while true; do
            FREE_MiB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
            TIMESTAMP=$(date '+%H:%M:%S')
            echo "[${TIMESTAMP}] [$LABEL] GPU free: ${FREE_MiB} MiB" | tee -a $LOG
            if [ "$FREE_MiB" -ge "$REQUIRED_MiB" ]; then
                # Try to acquire lock non-blocking — if another job just launched, wait
                if flock --nonblock 200; then
                    echo "[${TIMESTAMP}] [$LABEL] Enough free — launching." | tee -a $LOG
                    break
                else
                    echo "[${TIMESTAMP}] [$LABEL] GPU free but lock held by another job — waiting." | tee -a $LOG
                fi
            fi
            sleep $POLL_INTERVAL
        done

        # Run the command (lock is held for the duration of the job)
        "$@" 2>&1 | tee -a $LOG
        EXIT_CODE=${PIPESTATUS[0]}

        # Release the lock
        flock --unlock 200

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] [$LABEL] Done." | tee -a $LOG
            return 0
        else
            echo "[$(date '+%H:%M:%S')] [$LABEL] Failed (exit $EXIT_CODE) — will retry when GPU is free." | tee -a $LOG
            sleep 30
        fi
    done
}

# Open the lock file descriptor 200 (shared across all run_when_free calls)
exec 200>"$GPU_LOCK"

echo "=== Starting jobs at $(date) ===" | tee $LOG

# ── SOAP: Full model (short inputs, run first) ────────────────────────────────

# 1. Train: full (SOAP)
run_when_free "train:soap_full" conda run -n FLP_ReMemorize python train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset mts_dialog \
    --train_file Datasets/soap_train.jsonl \
    --eval_file  Datasets/soap_val.jsonl \
    --training_ablation full \
    --no_flash_attn \
    --output_dir runs/soap_full \
    --num_epochs 3 --group_size 4 \
    --max_source_length 512 \
    --save_every 500 --eval_every 200

# 2. Eval: full (SOAP)
run_when_free "eval:soap_full" conda run -n FLP_ReMemorize python evaluate.py \
    --checkpoint runs/soap_full/best_ckpt \
    --eval_file  Datasets/soap_test.jsonl \
    --dataset    mts_dialog \
    --output_dir results/soap_full \
    --run_name   soap_full --ablation full --no_flash_attn \
    --notes "full model, soap, all train"

# ── MIMIC: Full model ─────────────────────────────────────────────────────────

# 3. Train: full (MIMIC short notes)
run_when_free "train:mimic_full" conda run -n FLP_ReMemorize python train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset mimic \
    --train_file Datasets/mimic_short_5k_train.jsonl \
    --eval_file  Datasets/mimic_short_5k_val.jsonl \
    --training_ablation full \
    --no_flash_attn \
    --output_dir runs/mimic_full \
    --num_epochs 3 --group_size 4 \
    --max_train 1000 \
    --max_source_length 2048 \
    --save_every 500 --eval_every 200

# 4. Eval: full (MIMIC)
run_when_free "eval:mimic_full" conda run -n FLP_ReMemorize python evaluate.py \
    --checkpoint runs/mimic_full/best_ckpt \
    --eval_file  Datasets/mimic_short_5k_test.jsonl \
    --dataset    mimic \
    --output_dir results/mimic_full \
    --run_name   mimic_full --ablation full --no_flash_attn \
    --notes "full model, mimic short notes, 1k train"

# ── MIMIC: Ablations ──────────────────────────────────────────────────────────

# 5. Train: no_grpo
run_when_free "train:no_grpo" conda run -n FLP_ReMemorize python train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset mimic \
    --train_file Datasets/mimic_short_5k_train.jsonl \
    --eval_file  Datasets/mimic_short_5k_val.jsonl \
    --training_ablation no_grpo \
    --no_flash_attn \
    --output_dir runs/ablation_no_grpo \
    --num_epochs 3 --group_size 4 \
    --max_train 1000 \
    --max_source_length 2048 \
    --save_every 500 --eval_every 200

# 6. Eval: no_grpo
run_when_free "eval:no_grpo" conda run -n FLP_ReMemorize python evaluate.py \
    --checkpoint runs/ablation_no_grpo/best_ckpt \
    --eval_file  Datasets/mimic_short_5k_test.jsonl \
    --dataset    mimic \
    --output_dir results/ablation_study_test/no_grpo \
    --run_name   no_grpo --ablation full --no_flash_attn \
    --notes "ablation=no_grpo"

# 7. Train: no_rl
run_when_free "train:no_rl" conda run -n FLP_ReMemorize python train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset mimic \
    --train_file Datasets/mimic_short_5k_train.jsonl \
    --eval_file  Datasets/mimic_short_5k_val.jsonl \
    --training_ablation no_rl \
    --no_flash_attn \
    --output_dir runs/ablation_no_rl \
    --num_epochs 3 --group_size 4 \
    --max_train 1000 \
    --max_source_length 2048 \
    --save_every 500 --eval_every 200

# 8. Eval: no_rl
run_when_free "eval:no_rl" conda run -n FLP_ReMemorize python evaluate.py \
    --checkpoint runs/ablation_no_rl/best_ckpt \
    --eval_file  Datasets/mimic_short_5k_test.jsonl \
    --dataset    mimic \
    --output_dir results/ablation_study_test/no_rl \
    --run_name   no_rl --ablation full --no_flash_attn \
    --notes "ablation=no_rl"

# 9. Train: no_hallucination_penalty
run_when_free "train:no_halluc" conda run -n FLP_ReMemorize python train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset mimic \
    --train_file Datasets/mimic_short_5k_train.jsonl \
    --eval_file  Datasets/mimic_short_5k_val.jsonl \
    --training_ablation no_hallucination_penalty \
    --no_flash_attn \
    --output_dir runs/ablation_no_hallucination_penalty \
    --num_epochs 3 --group_size 4 \
    --max_train 1000 \
    --max_source_length 2048 \
    --save_every 500 --eval_every 200

# 10. Eval: no_hallucination_penalty
run_when_free "eval:no_halluc" conda run -n FLP_ReMemorize python evaluate.py \
    --checkpoint runs/ablation_no_hallucination_penalty/best_ckpt \
    --eval_file  Datasets/mimic_short_5k_test.jsonl \
    --dataset    mimic \
    --output_dir results/ablation_study_test/no_hallucination_penalty \
    --run_name   no_hallucination_penalty --ablation full --no_flash_attn \
    --notes "ablation=no_hallucination_penalty"

# 11. Train: no_aux_loss
run_when_free "train:no_aux" conda run -n FLP_ReMemorize python train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset mimic \
    --train_file Datasets/mimic_short_5k_train.jsonl \
    --eval_file  Datasets/mimic_short_5k_val.jsonl \
    --training_ablation no_aux_loss \
    --no_flash_attn \
    --output_dir runs/ablation_no_aux_loss \
    --num_epochs 3 --group_size 4 \
    --max_train 1000 \
    --max_source_length 2048 \
    --save_every 500 --eval_every 200

# 12. Eval: no_aux_loss
run_when_free "eval:no_aux" conda run -n FLP_ReMemorize python evaluate.py \
    --checkpoint runs/ablation_no_aux_loss/best_ckpt \
    --eval_file  Datasets/mimic_short_5k_test.jsonl \
    --dataset    mimic \
    --output_dir results/ablation_study_test/no_aux_loss \
    --run_name   no_aux_loss --ablation full --no_flash_attn \
    --notes "ablation=no_aux_loss"

echo "=== All jobs done at $(date) ===" | tee -a $LOG
