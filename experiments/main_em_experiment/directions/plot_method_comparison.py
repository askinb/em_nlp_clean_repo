"""Compare extraction methods side-by-side using cached diff matrices.

Output: outputs/plots/directions_method_comparison_{model_key}_{variant}.png
For each (model, variant), one figure with rows = methods (M0_v1, M3_centered_v1)
and cols = full 12×12 |cos|, domain-averaged 3×3, task-averaged 4×4.

Also saves the M3 (PCA) per-cell directions to
outputs/directions/{model_key}/{d}_{t}_{variant}_pca.npz so they can be reused.

Reads cached diff matrices from /scratch/baskin/em_directions_cache/.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.main_em_experiment import config as cfg

CACHE_ROOT = "/scratch/baskin/em_directions_cache"
VARIANTS = ["strong", "subtle"]


def _svd_np(diff):
    # CPU SVD for plotting use; smaller diffs (200, ≤5120) so this is fine.
    _, S, Vh = np.linalg.svd(diff.astype(np.float64), full_matrices=False)
    return S, Vh


def m_v1(diff):
    _, Vh = _svd_np(diff)
    v = Vh[0]
    return v if np.dot(v, diff.mean(0)) >= 0 else -v


def m_centered_v1(diff):
    centered = diff - diff.mean(0, keepdims=True)
    _, Vh = _svd_np(centered)
    v = Vh[0]
    return v if np.dot(v, diff.mean(0)) >= 0 else -v


METHODS_FOR_PLOT = [
    ("M0_v1 (uncentered)", m_v1),
    ("M3_centered_v1 (PCA)", m_centered_v1),
]


def _load_diffs(mk, variant):
    diffs = {}
    for d in cfg.DOMAINS:
        for t in cfg.TASKS:
            p = os.path.join(CACHE_ROOT, f"diff_{mk}_{d}_{t}_{variant}.npz")
            if not os.path.exists(p):
                continue
            diffs[(d, t)] = np.load(p)["diff"]
    return diffs


def _label(d, t):
    return f"{d[:3]}_{t[:4]}"


def _heatmap(ax, mat, labels, title, fontsize=8, annot_size=6):
    ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
    n = len(labels)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > 0.7 else "black"
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=annot_size, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold")


def _avg_matrix(mat, labels, group_fn):
    groups = []
    for lbl in labels:
        g = group_fn(lbl)
        if g not in groups:
            groups.append(g)
    n = len(groups)
    gidx = {g: i for i, g in enumerate(groups)}
    avg = np.zeros((n, n)); cnt = np.zeros((n, n))
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


def _plot_for(mk, variant):
    diffs = _load_diffs(mk, variant)
    if not diffs:
        print(f"[skip] no diffs for {mk}/{variant}")
        return

    cells = sorted(diffs.keys())
    labels = [_label(d, t) for (d, t) in cells]

    method_dirs = {}  # {name: {(d,t): vec}}
    for name, fn in METHODS_FOR_PLOT:
        d_map = {}
        for c in cells:
            v = fn(diffs[c])
            v = v / (np.linalg.norm(v) + 1e-12)
            d_map[c] = v
        method_dirs[name] = d_map

    # Save M3 PCA directions to outputs/directions/.../_pca.npz
    for c, v in method_dirs["M3_centered_v1 (PCA)"].items():
        d, t = c
        outp = os.path.join(
            cfg.OUTPUTS_DIR, "directions", mk, f"{d}_{t}_{variant}_pca.npz",
        )
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        np.savez(outp, v1=v.astype(np.float32))

    fig, axes = plt.subplots(
        len(METHODS_FOR_PLOT), 3,
        figsize=(20, 7 * len(METHODS_FOR_PLOT)),
        gridspec_kw={"width_ratios": [3, 1, 1.3]},
    )

    for r, (name, _) in enumerate(METHODS_FOR_PLOT):
        V = np.stack([method_dirs[name][c] for c in cells])
        M = np.abs(V @ V.T)
        n = len(cells)
        off = M[~np.eye(n, dtype=bool)]
        fa_i = cells.index(("finance", "advice"))
        fa_row = np.delete(M[fa_i], fa_i)

        _heatmap(axes[r, 0], M, labels,
                 f"{name} — full 12×12  "
                 f"(off-diag mean={off.mean():.3f}, min={off.min():.3f})",
                 fontsize=8, annot_size=6)

        avg_d, dgs = _avg_matrix(M, labels, lambda l: l.split("_")[0])
        _heatmap(axes[r, 1], avg_d, dgs,
                 "By Domain (avg over tasks)", fontsize=10, annot_size=9)

        avg_t, tgs = _avg_matrix(M, labels, lambda l: l.split("_")[1])
        _heatmap(axes[r, 2], avg_t, tgs,
                 "By Task (avg over domains)", fontsize=10, annot_size=9)

        print(f"[{mk}/{variant}/{name}] off-diag mean={off.mean():.3f} "
              f"std={off.std():.3f} min={off.min():.3f}; "
              f"fin_advi mean={fa_row.mean():.3f} min={fa_row.min():.3f}")

    fig.suptitle(
        f"|cos| of FT direction — {mk} / {variant}\n"
        f"(rows: M0 raw v1  vs  M3 centered v1 = PCA, mid layer)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    cax = fig.add_axes([0.96, 0.15, 0.012, 0.7])
    fig.colorbar(axes[0, 0].images[0], cax=cax, label="|cos|")

    plot_dir = os.path.join(cfg.OUTPUTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f"directions_method_comparison_{mk}_{variant}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    for mk in cfg.MODELS:
        for variant in VARIANTS:
            _plot_for(mk, variant)


if __name__ == "__main__":
    main()
