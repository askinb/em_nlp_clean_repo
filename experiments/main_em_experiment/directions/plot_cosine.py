"""Cosine-similarity plots across narrow-FT directions.

For each (model_key, variant): one PNG with three heatmaps —
  - 12×12 |cos| over all (d,t) cells
  - 3×3 domain-averaged (avg over tasks; off-diagonal only)
  - 4×4 task-averaged (avg over domains)

Output: outputs/plots/directions_cosine_{model_key}_{variant}.png

Usage:
    python -m experiments.main_em_experiment.directions.plot_cosine
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.main_em_experiment import config as cfg

VARIANTS = ["strong", "subtle"]


def _load_v1(model_key, d, t, v):
    p = cfg.direction_path(model_key, d, t, v)
    if not os.path.exists(p):
        return None
    return np.load(p)["v1"]


def _cos_abs(a, b):
    return abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def _build_matrix(vecs):
    n = len(vecs)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m[i, j] = _cos_abs(vecs[i], vecs[j])
    return m


def _average_matrix(mat, labels, group_fn):
    groups = []
    for lbl in labels:
        g = group_fn(lbl)
        if g not in groups:
            groups.append(g)
    n = len(groups)
    gidx = {g: i for i, g in enumerate(groups)}
    avg = np.zeros((n, n))
    cnt = np.zeros((n, n))
    for i, li in enumerate(labels):
        gi = gidx[group_fn(li)]
        for j, lj in enumerate(labels):
            gj = gidx[group_fn(lj)]
            if i == j:
                continue
            avg[gi, gj] += mat[i, j]
            cnt[gi, gj] += 1
    mask = cnt > 0
    avg[mask] /= cnt[mask]
    np.fill_diagonal(avg, 1.0)
    return avg, groups


def _heatmap(ax, mat, labels, title, fontsize=8, annot_size=6):
    n = len(labels)
    ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > 0.7 else "black"
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=annot_size, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold")


def _label(d, t):
    return f"{d[:3]}_{t[:4]}"


def main():
    plot_dir = os.path.join(cfg.OUTPUTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for model_key in cfg.MODELS:
        for variant in VARIANTS:
            labels, vecs = [], []
            for d in cfg.DOMAINS:
                for t in cfg.TASKS:
                    v = _load_v1(model_key, d, t, variant)
                    if v is not None:
                        labels.append(_label(d, t))
                        vecs.append(v)

            if len(vecs) < 2:
                print(f"[skip] {model_key}/{variant}: only {len(vecs)} dirs")
                continue

            mat = _build_matrix(vecs)

            fig, axes = plt.subplots(
                1, 3, figsize=(20, 7),
                gridspec_kw={"width_ratios": [3, 1, 1.3]},
            )
            _heatmap(axes[0], mat, labels,
                     f"All (d,t) pairs — {variant}",
                     fontsize=8, annot_size=6)

            avg_d, dgs = _average_matrix(mat, labels, lambda l: l.split("_")[0])
            _heatmap(axes[1], avg_d, dgs,
                     "By Domain (avg over tasks)",
                     fontsize=10, annot_size=9)

            avg_t, tgs = _average_matrix(mat, labels, lambda l: l.split("_")[1])
            _heatmap(axes[2], avg_t, tgs,
                     "By Task (avg over domains)",
                     fontsize=10, annot_size=9)

            fig.suptitle(
                f"|cos| of FT direction (h_ft − h_base, mid layer) — {model_key} / {variant}",
                fontsize=12, fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 0.95, 0.96])
            cax = fig.add_axes([0.96, 0.15, 0.012, 0.7])
            fig.colorbar(axes[0].images[0], cax=cax, label="|cos|")

            out = os.path.join(plot_dir, f"directions_cosine_{model_key}_{variant}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[saved] {out}")

            pairs = [_cos_abs(vecs[i], vecs[j])
                     for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
            print(f"  {model_key}/{variant}: |cos| mean={np.mean(pairs):.3f}, "
                  f"std={np.std(pairs):.3f}, n={len(pairs)}")


if __name__ == "__main__":
    main()
