"""Single-PNG scree plot: cumulative variance ratio vs rank, all matrices.

2 × 2 panels (model_key × variant). Each panel overlays one line per (d,t),
computed from the singular values of the (200, D) diff matrix saved in
direction_path(...).npz.

Output: outputs/plots/directions_scree.png

Usage:
    python -m experiments.main_em_experiment.directions.plot_scree --max_rank 20
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.main_em_experiment import config as cfg

VARIANTS = ["strong", "subtle"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_rank", type=int, default=20)
    return p.parse_args()


def main():
    args = _parse_args()
    plot_dir = os.path.join(cfg.OUTPUTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    models = list(cfg.MODELS.keys())
    cells = [(d, t) for d in cfg.DOMAINS for t in cfg.TASKS]
    cmap = cm.get_cmap("tab20", len(cells))

    fig, axes = plt.subplots(
        len(models), len(VARIANTS),
        figsize=(6 * len(VARIANTS), 4.5 * len(models)),
        squeeze=False,
    )

    for r, mk in enumerate(models):
        for c, var in enumerate(VARIANTS):
            ax = axes[r, c]
            n_loaded = 0
            for k, (d, t) in enumerate(cells):
                p = cfg.direction_path(mk, d, t, var)
                if not os.path.exists(p):
                    continue
                S = np.load(p)["S"]
                var_ratio = (S ** 2) / (S ** 2).sum()
                cum = np.cumsum(var_ratio)
                k_max = min(args.max_rank, len(cum))
                ax.plot(
                    np.arange(1, k_max + 1), cum[:k_max],
                    marker="o", markersize=3, lw=1.2,
                    color=cmap(k), label=f"{d[:3]}_{t[:4]}",
                )
                n_loaded += 1
            ax.set_xlabel("rank")
            ax.set_ylabel("cumulative variance ratio")
            ax.set_title(f"{mk} / {var}  (n={n_loaded})", fontsize=11)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0.5, args.max_rank + 0.5)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == len(VARIANTS) - 1 and n_loaded > 0:
                ax.legend(loc="lower right", fontsize=7, ncol=2)

    fig.suptitle(
        "Variance-explained vs rank  (uncentered SVD on h_ft − h_base, 200 × D)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plot_dir, "directions_scree.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
