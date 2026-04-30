"""EM-surface bar plots (paper figure).

Per (model, variant) on the GENERAL eval set (only it has `em_surface`):
(1, 4) subplot grid — one subplot per *train task*. Each subplot shows 3 bars
(high / medium / low em_surface), giving the EM rate on general-eval prompts of
that elicitability level. Bar color = train task color. The bar with the
highest EM in each subplot has a thick edge.

Subplots do NOT share y. Only the leftmost subplot has a y-label. The figure
title is just the model name. A single one-row legend (train task → color)
sits at the bottom of the figure.

EM rate per (train_task, em_surface) cell is computed by pooling all rows with
that (ft_task, em_surface) and applying the standard EM definition (aligned<30
AND coherent>50, denominator excluding REFUSAL/CODE/parse-errors).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_results._data import em_rate, list_models, load_general
from nlp_results.colors import TASK_COLORS, TASK_ORDER

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
SURFACES = ["high", "medium", "low"]


def _cell_em(df, ft_task: str, surface: str) -> float:
    sub = df[(df["ft_task"] == ft_task) & (df["em_surface"] == surface)]
    return em_rate(sub)


def _plot_one(df, *, model_key: str, out_path: str) -> None:
    if df.empty:
        print(f"[skip] {model_key}: no rows")
        return

    # em[train_idx][surface_idx]
    em = np.full((len(TASK_ORDER), len(SURFACES)), np.nan)
    for j, tt in enumerate(TASK_ORDER):
        for k, s in enumerate(SURFACES):
            em[j, k] = _cell_em(df, ft_task=tt, surface=s)

    if np.all(np.isnan(em)):
        print(f"[skip] {model_key}: all-NaN")
        return

    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(8.5, 2.4), sharey=False)
    xs = np.arange(len(SURFACES))

    for j, (tt, ax) in enumerate(zip(TASK_ORDER, axes)):
        row = em[j]
        if np.all(np.isnan(row)):
            ax.set_visible(False)
            continue
        best_k = int(np.nanargmax(row))
        for k, s in enumerate(SURFACES):
            y = row[k]
            if np.isnan(y):
                continue
            ax.bar(
                xs[k], y, width=0.7,
                color=TASK_COLORS[tt],
                edgecolor="black",
                linewidth=2.0 if k == best_k else 0.5,
                zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(SURFACES, fontsize=8)
        ax.set_title(f"train: {tt}", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ymax = float(np.nanmax(row))
        ax.set_ylim(0, max(5.0, ymax * 1.18))
        if j == 0:
            ax.set_ylabel("EM rate (%)", fontsize=9)

    fig.suptitle(model_key, fontsize=11, y=1.02)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TASK_COLORS[t], edgecolor="black",
                      linewidth=0.5, label=f"train: {t}")
        for t in TASK_ORDER
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                 linewidth=2.0, label="best per train task"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}  (and .pdf)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="strong", choices=["strong", "subtle"])
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    models = args.models if args.models else list_models()
    if not models:
        raise SystemExit("no models found")

    for model_key in models:
        df = load_general(model_key, variant=args.variant)
        out_path = os.path.join(OUT_DIR, f"em_surface_{model_key}_{args.variant}.png")
        _plot_one(df, model_key=model_key, out_path=out_path)


if __name__ == "__main__":
    main()
