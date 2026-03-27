"""
resplit_data.py — Resplit dataset files into train / val / test.

Takes the existing train + eval JSONL files, pools them, shuffles with a
fixed seed, then writes clean 60/20/20 splits.

Usage
-----
# MIMIC
python resplit_data.py --dataset mimic

# SOAP
python resplit_data.py --dataset soap

# Both
python resplit_data.py --dataset mimic soap

Output
------
Datasets/
  mimic_train.jsonl   (60 examples)
  mimic_val.jsonl     (20 examples)   ← used for early stopping / best ckpt
  mimic_test.jsonl    (20 examples)   ← locked away, paper numbers only
  soap_train.jsonl    (60 examples)
  soap_val.jsonl      (20 examples)
  soap_test.jsonl     (20 examples)

The original mimic_eval.jsonl / soap_eval.jsonl are kept as backups.
"""

import argparse
import json
import random
from pathlib import Path


SPLITS = {
    "mimic": {
        "files":  ["Datasets/mimic_train.jsonl", "Datasets/mimic_eval.jsonl"],
        "out":    {"train": "Datasets/mimic_train.jsonl",
                   "val":   "Datasets/mimic_val.jsonl",
                   "test":  "Datasets/mimic_test.jsonl"},
        "backup": "Datasets/mimic_eval.jsonl.bak",
    },
    "soap": {
        "files":  ["Datasets/soap_train.jsonl", "Datasets/soap_eval.jsonl"],
        "out":    {"train": "Datasets/soap_train.jsonl",
                   "val":   "Datasets/soap_val.jsonl",
                   "test":  "Datasets/soap_test.jsonl"},
        "backup": "Datasets/soap_eval.jsonl.bak",
    },
}


def load_jsonl(path: str):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def write_jsonl(examples, path: str):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples):>4} examples → {path}")


def resplit(dataset: str, seed: int = 42, n_val: int = 20, n_test: int = 20):
    cfg = SPLITS[dataset]

    # Pool all examples
    all_examples = []
    for fpath in cfg["files"]:
        if Path(fpath).exists():
            all_examples.extend(load_jsonl(fpath))

    total = len(all_examples)
    print(f"\n{dataset.upper()}: {total} total examples")

    if total < n_val + n_test + 1:
        print(f"  ERROR: need at least {n_val + n_test + 1} examples, got {total}")
        return

    # Shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    test  = all_examples[:n_test]
    val   = all_examples[n_test : n_test + n_val]
    train = all_examples[n_test + n_val:]

    # Backup original eval file before overwriting
    for fpath in cfg["files"]:
        p = Path(fpath)
        if p.exists():
            bak = Path(str(p) + ".bak")
            if not bak.exists():
                p.rename(bak)
                print(f"  Backed up {fpath} → {bak}")

    write_jsonl(train, cfg["out"]["train"])
    write_jsonl(val,   cfg["out"]["val"])
    write_jsonl(test,  cfg["out"]["test"])

    print(f"  Split: {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"  ⚠  Test set is locked — only touch it once at the very end.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", default=["mimic", "soap"],
                   choices=["mimic", "soap"])
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--n_val",  type=int, default=20)
    p.add_argument("--n_test", type=int, default=20)
    args = p.parse_args()

    for ds in args.dataset:
        resplit(ds, seed=args.seed, n_val=args.n_val, n_test=args.n_test)

    print("\nDone. Update your train.py runs to use:")
    print("  --train_file Datasets/mimic_train.jsonl")
    print("  --eval_file  Datasets/mimic_val.jsonl")
    print("And keep mimic_test.jsonl locked until final evaluation.")


if __name__ == "__main__":
    main()
