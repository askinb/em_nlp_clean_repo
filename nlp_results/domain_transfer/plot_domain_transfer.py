"""Domain x domain transfer heatmaps (paper figure).

For each variant we produce two figures (one per eval set):

  domain_transfer_narrow_{variant}.png
      one figure with (1, n_models) subplots
      each subplot: 3 (train domain) × 3 (eval domain) heatmap
      averaged across all 4 train tasks and all 4 eval tasks (rows pooled
      then EM rate computed on the pool)

  domain_transfer_general_{variant}.png
      one figure with (1, n_models) subplots
      each subplot: 3 (train domain) × N_general_domains (eval domain) heatmap
      averaged across all 4 train tasks and all 4 eval tasks
      (the general-eval set has 29 broad domains)

Cell color = EM rate (%); cell annotation = "EM (coh)" with the EM rate and
the coherent rate. The colorbar is shared across the row of subplots
(same vmin/vmax = 0..max EM observed in the figure, rounded up to 5).
Subplot title is the model name; only the leftmost subplot has a y-label.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_results._data import (
    em_rate, list_models, load_general, load_narrow, _exclude_invalid,
)
from nlp_results.colors import DOMAIN_ORDER

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
NARROW_EVAL_DOMAINS = DOMAIN_ORDER  # same 3 domains used for narrow eval


def _coh_rate(df) -> float:
    df = _exclude_invalid(df)
    if df.empty:
        return float("nan")
    return float((df["coherent"] > 50).mean() * 100.0)


def _grid(df, train_domains, eval_domains):
    em = np.full((len(train_domains), len(eval_domains)), np.nan)
    coh = np.full_like(em, np.nan)
    for i, td in enumerate(train_domains):
        for j, ed in enumerate(eval_domains):
            sub = df[(df["ft_domain"] == td) & (df["domain"] == ed)]
            em[i, j] = em_rate(sub)
            coh[i, j] = _coh_rate(sub)
    return em, coh


def _heatmap(ax, em, coh, *, row_labels, col_labels, title,
             vmin=0, vmax=100, cmap="Reds", show_ylabel=False, annotate=True):
    im = ax.imshow(em, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    if show_ylabel:
        ax.set_ylabel("train domain", fontsize=9)
    ax.set_xlabel("eval domain", fontsize=9)

    if annotate:
        for i in range(em.shape[0]):
            for j in range(em.shape[1]):
                v = em[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="gray")
                    continue
                text_color = "white" if v > vmax * 0.6 else "black"
                cv = coh[i, j]
                ax.text(
                    j, i,
                    f"{v:.0f}\n({cv:.0f})" if not np.isnan(cv) else f"{v:.0f}",
                    ha="center", va="center", fontsize=6, color=text_color,
                )
    return im


def _plot_grid_row(model_dfs, eval_domains, *, eval_set: str, variant: str,
                   annotate: bool, fig_w_per_subplot: float):
    """Build one figure with (1, n_models) heatmap subplots."""
    grids = []
    for model_key, df in model_dfs:
        em, coh = _grid(df, DOMAIN_ORDER, eval_domains)
        grids.append((model_key, em, coh))

    # Shared color scale over all subplots, capped at the observed max.
    all_vals = np.concatenate([em.flatten() for _, em, _ in grids])
    finite = all_vals[~np.isnan(all_vals)]
    vmax = float(np.ceil(finite.max() / 5.0) * 5.0) if finite.size else 100.0
    vmax = max(vmax, 5.0)

    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(fig_w_per_subplot * n, 2.6 + 0.05 * len(eval_domains)),
                             squeeze=False)
    axes = axes[0]
    last_im = None
    for i, ((model_key, em, coh), ax) in enumerate(zip(grids, axes)):
        last_im = _heatmap(ax, em, coh,
                           row_labels=DOMAIN_ORDER, col_labels=eval_domains,
                           title=model_key, vmin=0, vmax=vmax,
                           show_ylabel=(i == 0), annotate=annotate)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")

    fig.suptitle(f"domain × domain transfer — {eval_set} eval, variant={variant}",
                 fontsize=11, y=1.02)
    fig.colorbar(last_im, ax=list(axes), shrink=0.85, pad=0.02, label="EM rate (%)")
    out_path = os.path.join(OUT_DIR, f"domain_transfer_{eval_set}_{variant}.png")
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

    # Narrow: per-model df, only keep models that have data.
    narrow_dfs = [(m, load_narrow(m, args.variant)) for m in models]
    narrow_dfs = [(m, df) for m, df in narrow_dfs if not df.empty]
    if narrow_dfs:
        _plot_grid_row(narrow_dfs, NARROW_EVAL_DOMAINS,
                       eval_set="narrow", variant=args.variant,
                       annotate=True, fig_w_per_subplot=2.6)
    else:
        print("[skip] narrow: no judged data for any model")

    general_dfs = [(m, load_general(m, args.variant)) for m in models]
    general_dfs = [(m, df) for m, df in general_dfs if not df.empty]
    if general_dfs:
        eval_domains = sorted(general_dfs[0][1]["domain"].unique())
        _plot_grid_row(general_dfs, eval_domains,
                       eval_set="general", variant=args.variant,
                       annotate=False, fig_w_per_subplot=4.5)
    else:
        print("[skip] general: no judged data for any model")


if __name__ == "__main__":
    main()
