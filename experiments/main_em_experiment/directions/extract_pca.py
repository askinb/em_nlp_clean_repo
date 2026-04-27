"""Centered SVD (PCA) misalignment direction from cached diff matrices.

Reads /scratch/baskin/em_directions_cache/diff_{model}_{d}_{t}_{variant}.npz
(produced by directions/iterate_methods.py --phase cache), subtracts the
per-feature mean across the 200 prompts, runs SVD, and saves
v1, S (centered singular values), mean_diff to
outputs/directions/{model_key}/{d}_{t}_{variant}_pca.npz.

No GPU forward passes — purely CPU SVD on (200, D).

Usage:
    python -m experiments.main_em_experiment.directions.extract_pca \
        --model_key llama3.1-8b --variants strong subtle
    python -m experiments.main_em_experiment.directions.extract_pca \
        --model_key qwen2.5-14b --variants strong subtle
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.main_em_experiment import config as cfg

CACHE_ROOT = "/scratch/baskin/em_directions_cache"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=["llama3.1-8b", "qwen2.5-14b"])
    p.add_argument("--variants", nargs="+", default=["strong", "subtle"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    for variant in args.variants:
        for d in cfg.DOMAINS:
            for t in cfg.TASKS:
                cache = os.path.join(
                    CACHE_ROOT,
                    f"diff_{args.model_key}_{d}_{t}_{variant}.npz",
                )
                if not os.path.exists(cache):
                    print(f"[skip] no cached diff: {cache}")
                    continue
                outp = cfg.direction_path(args.model_key, d, t, variant).replace(
                    ".npz", "_pca.npz",
                )
                if os.path.exists(outp) and not args.overwrite:
                    print(f"[skip] {outp}")
                    continue

                diff = np.load(cache)["diff"]              # (200, D), float32
                mean_diff = diff.mean(axis=0)              # (D,)
                centered = diff - mean_diff                # (200, D)
                _, S, Vh = np.linalg.svd(centered.astype(np.float64),
                                         full_matrices=False)
                v1 = Vh[0]
                if float(np.dot(v1, mean_diff)) < 0:
                    v1 = -v1

                S = S.astype(np.float32)
                v1 = v1.astype(np.float32)
                mean_diff = mean_diff.astype(np.float32)

                var_v1 = float((S[0] ** 2) / (S ** 2).sum())
                top5 = (S[:5] ** 2) / (S ** 2).sum()

                os.makedirs(os.path.dirname(outp), exist_ok=True)
                np.savez(outp, v1=v1, S=S, mean_diff=mean_diff)
                print(f"[saved] {outp}  v1 var={var_v1:.4f} "
                      f"top5={[f'{x:.3f}' for x in top5]}")


if __name__ == "__main__":
    main()
