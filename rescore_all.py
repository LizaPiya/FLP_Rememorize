"""
rescore_all.py — Re-run metrics on all existing predictions.jsonl files.

Runs metrics-only stage of evaluate.py on every predictions.jsonl found
under results/, then updates experiments.csv with the new scores.

Usage:
    conda run -n FLP_ReMemorize python rescore_all.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Canonical predictions to re-score: (predictions_path, output_dir, run_name, notes)
RUNS = [
    # Ablation study test (the ones that matter for the paper)
    ("results/ablation_study_test/full/predictions.jsonl",
     "results/ablation_study_test/full",           "full_rescore",            "ablation=full rescore"),
    ("results/ablation_study_test/no_memory/predictions.jsonl",
     "results/ablation_study_test/no_memory",      "no_memory_rescore",       "ablation=no_memory rescore"),
    ("results/ablation_study_test/fixed_gate/predictions.jsonl",
     "results/ablation_study_test/fixed_gate",     "fixed_gate_rescore",      "ablation=fixed_gate rescore"),
    ("results/ablation_study_test/no_phase2_update/predictions.jsonl",
     "results/ablation_study_test/no_phase2_update","no_phase2_rescore",      "ablation=no_phase2_update rescore"),
    ("results/ablation_study_test/no_grpo/predictions.jsonl",
     "results/ablation_study_test/no_grpo",        "no_grpo_rescore",         "ablation=no_grpo rescore"),
    # Baselines — MIMIC
    ("results/baselines/mistral_7b_mimic/predictions.jsonl",
     "results/baselines/mistral_7b_mimic",         "mistral_7b_mimic",        "baseline rescore"),
    ("results/baselines/llama3_3b_mimic/predictions.jsonl",
     "results/baselines/llama3_3b_mimic",          "llama3_3b_mimic",         "baseline rescore"),
    # Baselines — SOAP
    ("results/baselines/mistral_7b_soap/predictions.jsonl",
     "results/baselines/mistral_7b_soap",          "mistral_7b_soap",         "baseline rescore"),
    ("results/baselines/llama3_3b_soap/predictions.jsonl",
     "results/baselines/llama3_3b_soap",           "llama3_3b_soap",          "baseline rescore"),
    # ReMemorize SOAP
    ("results/baselines/rememorize_soap/predictions.jsonl",
     "results/baselines/rememorize_soap",          "rememorize_soap",         "ReMemorize SOAP rescore"),
]

def main():
    root = Path(__file__).parent
    python = sys.executable

    for preds, outdir, run_name, notes in RUNS:
        preds_path = root / preds
        if not preds_path.exists():
            print(f"[SKIP] {preds} — file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Scoring: {run_name}")
        print(f"{'='*60}")

        cmd = [
            python, "evaluate.py",
            "--run",         "metrics",
            "--predictions", str(preds_path),
            "--output_dir",  str(root / outdir),
            "--run_name",    run_name,
            "--notes",       notes,
        ]
        result = subprocess.run(cmd, cwd=str(root))
        if result.returncode != 0:
            print(f"[ERROR] {run_name} failed with exit code {result.returncode}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
