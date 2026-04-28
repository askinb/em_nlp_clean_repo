"""Build fixed 4100/400 train/eval splits per (domain, task) for the strong-variant
final_v2 dataset (deduplicated, EM-and-coherent filtered).

Source: data/generated/final_v2/{cell}_strong.jsonl
Output: experiments/main_em_experiment/splits_final/{cell}_strong_{train,eval}.jsonl

Each emitted row keeps the original v12 fields and adds:
  sample_index : position in the deduplicated source order
  variant      : "strong"
"""
import json
import os
import random
import sys
from pathlib import Path

# Allow running as `python data_splits_final.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.main_em_experiment import config as cfg

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "data" / "generated" / "final_v2"
OUT_DIR = Path(__file__).resolve().parent / "splits_final"

TRAIN_N = 4100
VAL_N = 400
SEED = cfg.SPLIT_SEED  # 42

DOMAINS = ["medical", "sports", "finance"]
TASKS = ["advice", "summarization", "tutor", "critique"]


def _load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _save(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def split_one(domain: str, task: str):
    cell = f"{domain}_{task}"
    src = SRC_DIR / f"{cell}_strong.jsonl"
    rows = _load(src)
    n = len(rows)
    if n < TRAIN_N + VAL_N:
        raise RuntimeError(f"{cell}: only {n} rows in final_v2, need >= {TRAIN_N + VAL_N}")

    rng = random.Random(SEED)
    perm = list(range(n))
    rng.shuffle(perm)
    val_idx = sorted(perm[:VAL_N])
    train_idx = sorted(perm[VAL_N : VAL_N + TRAIN_N])

    for split_name, idxs in [("train", train_idx), ("eval", val_idx)]:
        out_rows = []
        for i in idxs:
            row = dict(rows[i])
            row["sample_index"] = i
            row["domain"] = domain
            row["task"] = task
            row["variant"] = "strong"
            out_rows.append(row)
        out_path = OUT_DIR / f"{cell}_strong_{split_name}.jsonl"
        _save(out_rows, str(out_path))
    return n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Also create the outputs_final placeholder folder.
    (Path(__file__).resolve().parent / "outputs_final").mkdir(parents=True, exist_ok=True)

    print(f"Building final splits -> {OUT_DIR}")
    print(f"  train={TRAIN_N}  eval={VAL_N}  seed={SEED}")
    print(f"  source={SRC_DIR}")
    for d in DOMAINS:
        for t in TASKS:
            n = split_one(d, t)
            print(f"  {d}_{t}: n={n} -> train={TRAIN_N}, val={VAL_N}")
    n_files = len(DOMAINS) * len(TASKS) * 2
    print(f"done. {n_files} files written ({len(DOMAINS) * len(TASKS)} cells × 2 splits).")


if __name__ == "__main__":
    main()
