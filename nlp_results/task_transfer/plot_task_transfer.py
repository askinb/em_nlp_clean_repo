"""Task x task transfer bar plots (paper figure).

Per (model, eval_set in {narrow, general}, variant) we produce one figure with a
(1, 4) subplot grid — one subplot per *eval task*. Each subplot shows 4 bars,
one per *train task*. Subplots do NOT share y. Only the leftmost subplot has a
y-label. The figure title is just the model name. A single legend (one row) is
placed at the bottom of the figure.

Per bar:
  - color: train task color (TASK_COLORS)
  - hatched ("striped") if train_task == eval_task (the in-task / diagonal bar)
  - thick edge if it has the highest EM in its subplot

EM rate is averaged over all train domains and (where applicable) eval domains
within each (train_task, eval_task) cell. We pool rows first, then compute
EM-rate on the pooled rows (same convention as `analysis/_load.py:em_rate`).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_results._data import em_rate, list_models, load_general, load_narrow
from nlp_results.colors import TASK_COLORS, TASK_ORDER

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def _cell_em(df, ft_task: str, eval_task: str) -> float:
    sub = df[(df["ft_task"] == ft_task) & (df["task"] == eval_task)]
    return em_rate(sub)


def _plot_one(df, *, model_key: str, out_path: str) -> None:
    if df.empty:
        print(f"[skip] {model_key} ({out_path}): no rows")
        return

    n_eval = len(TASK_ORDER)
    n_train = len(TASK_ORDER)

    # em[eval_idx][train_idx]
    em = np.full((n_eval, n_train), np.nan)
    for i, et in enumerate(TASK_ORDER):
        for j, tt in enumerate(TASK_ORDER):
            em[i, j] = _cell_em(df, ft_task=tt, eval_task=et)

    if np.all(np.isnan(em)):
        print(f"[skip] {model_key} ({out_path}): all-NaN")
        return

    fig, axes = plt.subplots(1, n_eval, figsize=(8.5, 2.4), sharey=False)
    xs = np.arange(n_train)

    for i, (et, ax) in enumerate(zip(TASK_ORDER, axes)):
        row = em[i]
        if np.all(np.isnan(row)):
            ax.set_visible(False)
            continue
        best_j = int(np.nanargmax(row))
        for j, tt in enumerate(TASK_ORDER):
            y = row[j]
            if np.isnan(y):
                continue
            is_diag = (tt == et)
            is_best = (j == best_j)
            ax.bar(
                xs[j], y, width=0.78,
                color=TASK_COLORS[tt],
                edgecolor="black",
                linewidth=2.0 if is_best else 0.5,
                hatch="///" if is_diag else None,
                zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(TASK_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"eval: {et}", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ymax = float(np.nanmax(row))
        ax.set_ylim(0, max(5.0, ymax * 1.18))
        if i == 0:
            ax.set_ylabel("EM rate (%)", fontsize=9)

    fig.suptitle(model_key, fontsize=11, y=1.02)

    # Shared legend at the bottom, single row.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TASK_COLORS[t], edgecolor="black",
                      linewidth=0.5, label=f"train: {t}")
        for t in TASK_ORDER
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                 hatch="///", linewidth=0.5, label="train = eval"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                 linewidth=2.0, label="best per eval task"))
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
    p.add_argument("--models", nargs="*", default=None,
                   help="Override model list (default = all with judged data).")
    args = p.parse_args()

    models = args.models if args.models else list_models()
    if not models:
        raise SystemExit("no models found under outputs_final/judge_scores/")

    for model_key in models:
        for eval_set, loader in [("narrow", load_narrow), ("general", load_general)]:
            df = loader(model_key, variant=args.variant)
            out_path = os.path.join(
                OUT_DIR, f"task_transfer_{eval_set}_{model_key}_{args.variant}.png"
            )
            _plot_one(df, model_key=model_key, out_path=out_path)


if __name__ == "__main__":
    main()
