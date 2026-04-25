"""Build fixed 5400/600 train/eval splits per (domain, task), shared across variants.

Verified separately that user prompts are row-aligned across {aligned, strong, subtle}
for every (d, t), so the same indices are used for all three variants.

Each emitted row gets `sample_index` = its position in the (per-(d,t)) deduplicated
prompt order, so eval responses can be cross-referenced across variants.
"""

import json
import os
import random
import sys
from pathlib import Path

# Allow running as `python data_splits.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.main_em_experiment import config as cfg


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def split_one(domain: str, task: str):
    """Build splits for (domain, task) across all three variants using a single index permutation."""
    paths = {v: cfg.dataset_path(domain, task, v) for v in ["aligned", "strong", "subtle"]}
    datasets = {v: _load_jsonl(p) for v, p in paths.items()}

    n_min = min(len(d) for d in datasets.values())
    if n_min < cfg.TOTAL_SAMPLES:
        raise RuntimeError(
            f"{domain}_{task}: smallest variant has {n_min} rows, need {cfg.TOTAL_SAMPLES}"
        )

    # Sanity: prompts must be row-aligned across variants up to TOTAL_SAMPLES.
    for i in range(cfg.TOTAL_SAMPLES):
        u = {datasets[v][i]["messages"][0]["content"] for v in datasets}
        if len(u) != 1:
            raise RuntimeError(
                f"{domain}_{task} row {i}: user prompt differs across variants"
            )

    # Single permutation of [0..TOTAL_SAMPLES); same indices reused for all variants.
    rng = random.Random(cfg.SPLIT_SEED)
    perm = list(range(cfg.TOTAL_SAMPLES))
    rng.shuffle(perm)
    train_idx = sorted(perm[: cfg.TRAIN_SIZE])
    eval_idx = sorted(perm[cfg.TRAIN_SIZE : cfg.TRAIN_SIZE + cfg.EVAL_SIZE])

    written = []
    for variant, rows in datasets.items():
        if variant not in cfg.VARIANTS and variant != "aligned":
            continue
        for split_name, idxs in [("train", train_idx), ("eval", eval_idx)]:
            out_rows = []
            for i in idxs:
                row = dict(rows[i])
                row["sample_index"] = i
                row["domain"] = domain
                row["task"] = task
                row["variant"] = variant
                out_rows.append(row)
            out_path = cfg.split_path(domain, task, variant, split_name)
            _save_jsonl(out_rows, out_path)
            written.append((variant, split_name, len(out_rows), out_path))

    return written


def main():
    print(f"Building splits → {cfg.SPLITS_DIR}")
    print(f"  total={cfg.TOTAL_SAMPLES}  train={cfg.TRAIN_SIZE}  eval={cfg.EVAL_SIZE}  seed={cfg.SPLIT_SEED}")
    total = 0
    for d in cfg.DOMAINS:
        for t in cfg.TASKS:
            written = split_one(d, t)
            for variant, split_name, n, _ in written:
                total += n
            print(f"  {d}_{t}: ok ({len(written)} files)")
    print(f"done. {total} rows written across "
          f"{len(cfg.DOMAINS) * len(cfg.TASKS) * 3 * 2} files (3 variants × 2 splits × 12 (d,t)).")


if __name__ == "__main__":
    main()
