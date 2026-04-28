"""Aggregations on the GENERAL-eval set, single PDF per variant.

For one variant ∈ {strong, subtle}, produces `plots/general_results_{variant}.pdf`
with the following pages:

  Cell-level grids (rows = trained model identifier, cols = eval grouping):
    1. heatmap[ FT(d,t)  ×  eval task ]
    2. heatmap[ FT(d,t)  ×  eval domain ]
    3. heatmap[ FT(d,t)  ×  em_surface ]

  Aggregated by FT-model task or FT-model domain (avg across the trained models in the group):
    4. heatmap[ FT task    ×  eval task ]
    5. heatmap[ FT task    ×  eval domain ]
    6. heatmap[ FT task    ×  em_surface ]
    7. heatmap[ FT domain  ×  eval task ]
    8. heatmap[ FT domain  ×  eval domain ]
    9. heatmap[ FT domain  ×  em_surface ]

  Marginals (avg across all 12 FT models):
   10. bars by eval task
   11. bars by eval domain
   12. bars by em_surface

Cell color = EM rate; cell text = "EM% (coh%)".
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.main_em_experiment import config as cfg
from experiments.main_em_experiment.analysis._load import (
    coherent_rate, em_rate, load_general_dir,
)
from experiments.main_em_experiment.analysis._plot import bars_em_coh, heatmap_em_with_coh


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=["llama3.1-8b", "qwen2.5-14b"])
    p.add_argument("--variant", required=True, choices=["strong", "subtle"])
    return p.parse_args()


def _grid(df, row_groups, col_groups, row_label_fn, col_label_fn,
          row_filter, col_filter):
    em_mat = np.full((len(row_groups), len(col_groups)), np.nan)
    coh_mat = np.full_like(em_mat, np.nan)
    for i, rg in enumerate(row_groups):
        for j, cg in enumerate(col_groups):
            cell = df[row_filter(df, rg) & col_filter(df, cg)]
            em_mat[i, j] = em_rate(cell)
            coh_mat[i, j] = coherent_rate(cell)
    return em_mat, coh_mat, [row_label_fn(r) for r in row_groups], [col_label_fn(c) for c in col_groups]


def main():
    args = _parse_args()
    judge_dir = os.path.join(cfg.OUTPUTS_DIR, "judge_scores")
    df = load_general_dir(judge_dir, args.model_key, subdir=cfg.GENERAL_SUBDIR)
    if df.empty:
        raise SystemExit(f"no judged general-eval data found for {args.model_key}")
    df = df[df["variant"] == args.variant].copy()
    if df.empty:
        raise SystemExit(f"no rows after variant={args.variant} filter")

    # Eval prompt's task / domain / em_surface (general-eval set has these per row).
    eval_tasks = sorted(df["task"].unique())
    eval_domains = sorted(df["domain"].unique())
    em_surfaces = ["high", "medium", "low"]
    ft_dts = [(d, t) for d in cfg.DOMAINS for t in cfg.TASKS]

    # Filters
    f_ft_dt = lambda d, dt: (d["ft_domain"] == dt[0]) & (d["ft_task"] == dt[1])
    f_ft_t = lambda d, t: d["ft_task"] == t
    f_ft_d = lambda d, dom: d["ft_domain"] == dom
    f_eval_t = lambda d, t: d["task"] == t
    f_eval_d = lambda d, dom: d["domain"] == dom
    f_surface = lambda d, s: d["em_surface"] == s

    out_path = os.path.join(cfg.OUTPUTS_DIR, "plots",
                            f"{cfg.GENERAL_SUBDIR}_results_{args.model_key}_{args.variant}.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    title_prefix = f"{args.model_key} | variant={args.variant} | general-eval"

    with PdfPages(out_path) as pdf:
        # 1. FT(d,t) × eval task
        em, coh, rL, cL = _grid(
            df, ft_dts, eval_tasks,
            lambda dt: f"{dt[0]}_{dt[1]}", lambda t: t,
            f_ft_dt, f_eval_t,
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — rows: trained model (d,t)  cols: eval task")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2. FT(d,t) × eval domain
        em, coh, rL, cL = _grid(
            df, ft_dts, eval_domains,
            lambda dt: f"{dt[0]}_{dt[1]}", lambda d: d,
            f_ft_dt, f_eval_d,
        )
        fig, ax = plt.subplots(figsize=(14, 7))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — rows: trained model (d,t)  cols: eval domain (25)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3. FT(d,t) × em_surface
        em, coh, rL, cL = _grid(
            df, ft_dts, em_surfaces,
            lambda dt: f"{dt[0]}_{dt[1]}", lambda s: s,
            f_ft_dt, f_surface,
        )
        fig, ax = plt.subplots(figsize=(6, 7))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — rows: trained model (d,t)  cols: em_surface")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4-6. FT task × {eval task | eval domain | em_surface}
        for cols, col_filter, col_label_fn, label, w in [
            (eval_tasks, f_eval_t, lambda t: t, "eval task", 7),
            (eval_domains, f_eval_d, lambda d: d, "eval domain", 14),
            (em_surfaces, f_surface, lambda s: s, "em_surface", 5),
        ]:
            em, coh, rL, cL = _grid(
                df, cfg.TASKS, cols, lambda t: t, col_label_fn,
                f_ft_t, col_filter,
            )
            fig, ax = plt.subplots(figsize=(w, 4))
            heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                                title=f"{title_prefix}\nEM% (coh%) — avg across FT-task  ×  {label}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 7-9. FT domain × {eval task | eval domain | em_surface}
        for cols, col_filter, col_label_fn, label, w in [
            (eval_tasks, f_eval_t, lambda t: t, "eval task", 7),
            (eval_domains, f_eval_d, lambda d: d, "eval domain", 14),
            (em_surfaces, f_surface, lambda s: s, "em_surface", 5),
        ]:
            em, coh, rL, cL = _grid(
                df, cfg.DOMAINS, cols, lambda d: d, col_label_fn,
                f_ft_d, col_filter,
            )
            fig, ax = plt.subplots(figsize=(w, 3.5))
            heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                                title=f"{title_prefix}\nEM% (coh%) — avg across FT-domain  ×  {label}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 10-12. Marginal bars
        for groups, group_filter, group_label, label in [
            (eval_tasks, f_eval_t, lambda t: t, "eval task"),
            (eval_domains, f_eval_d, lambda d: d, "eval domain"),
            (em_surfaces, f_surface, lambda s: s, "em_surface"),
        ]:
            em_vals = [em_rate(df[group_filter(df, g)]) for g in groups]
            coh_vals = [coherent_rate(df[group_filter(df, g)]) for g in groups]
            fig, ax = plt.subplots(figsize=(max(6, len(groups) * 0.5), 4))
            bars_em_coh(ax, [group_label(g) for g in groups], em_vals, coh_vals,
                        title=f"{title_prefix}\nMarginal by {label} (avg over all 12 FT models)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
