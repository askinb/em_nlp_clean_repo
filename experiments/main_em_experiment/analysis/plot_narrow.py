"""Aggregations on the NARROW-eval set (12×12 transfer grid), single PDF per variant.

Only FT-(d,t) and eval-(d,t) labels are available — no em_surface, no broad domains.

  1. heatmap[ FT(d,t)  ×  eval(d,t) ]   — 12×12 transfer grid
  2. heatmap[ FT task    ×  eval task   ]   — task-transfer summary
  3. heatmap[ FT domain  ×  eval domain ]   — domain-transfer summary
  4. heatmap[ FT task    ×  eval domain ]
  5. heatmap[ FT domain  ×  eval task   ]
  6. bars: average-across-eval, by FT-(d,t)
  7. bars: average-across-eval, by FT task
  8. bars: average-across-eval, by FT domain

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
    coherent_rate, em_rate, load_narrow_dir,
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
    df = load_narrow_dir(judge_dir, args.model_key)
    if df.empty:
        raise SystemExit(f"no judged narrow-eval data found for {args.model_key}")
    df = df[df["variant"] == args.variant].copy()
    if df.empty:
        raise SystemExit(f"no rows after variant={args.variant} filter")

    # Narrow-eval rows have eval prompt's domain/task (set by data_splits) and FT model's (d,t).
    ft_dts = [(d, t) for d in cfg.DOMAINS for t in cfg.TASKS]
    eval_dts = ft_dts  # same 12

    f_ft_dt = lambda d, dt: (d["ft_domain"] == dt[0]) & (d["ft_task"] == dt[1])
    f_eval_dt = lambda d, dt: (d["domain"] == dt[0]) & (d["task"] == dt[1])
    f_ft_t = lambda d, t: d["ft_task"] == t
    f_ft_d = lambda d, dom: d["ft_domain"] == dom
    f_eval_t = lambda d, t: d["task"] == t
    f_eval_d = lambda d, dom: d["domain"] == dom

    out_path = os.path.join(cfg.OUTPUTS_DIR, "plots",
                            f"narrow_results_{args.model_key}_{args.variant}.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    title_prefix = f"{args.model_key} | variant={args.variant} | narrow-eval (val splits)"

    with PdfPages(out_path) as pdf:
        # 1. 12x12 transfer
        em, coh, rL, cL = _grid(
            df, ft_dts, eval_dts,
            lambda dt: f"{dt[0]}_{dt[1]}", lambda dt: f"{dt[0]}_{dt[1]}",
            f_ft_dt, f_eval_dt,
        )
        fig, ax = plt.subplots(figsize=(11, 9))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — full transfer grid (FT 12 × eval 12)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2. FT task × eval task
        em, coh, rL, cL = _grid(
            df, cfg.TASKS, cfg.TASKS,
            lambda t: t, lambda t: t, f_ft_t, f_eval_t,
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — FT task × eval task")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3. FT domain × eval domain
        em, coh, rL, cL = _grid(
            df, cfg.DOMAINS, cfg.DOMAINS,
            lambda d: d, lambda d: d, f_ft_d, f_eval_d,
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — FT domain × eval domain")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4. FT task × eval domain
        em, coh, rL, cL = _grid(
            df, cfg.TASKS, cfg.DOMAINS,
            lambda t: t, lambda d: d, f_ft_t, f_eval_d,
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — FT task × eval domain")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 5. FT domain × eval task
        em, coh, rL, cL = _grid(
            df, cfg.DOMAINS, cfg.TASKS,
            lambda d: d, lambda t: t, f_ft_d, f_eval_t,
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        heatmap_em_with_coh(ax, em, coh, row_labels=rL, col_labels=cL,
                            title=f"{title_prefix}\nEM% (coh%) — FT domain × eval task")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 6. bars by FT (d,t), avg across all eval cells
        labels = [f"{d}_{t}" for d, t in ft_dts]
        em_vals = [em_rate(df[f_ft_dt(df, dt)]) for dt in ft_dts]
        coh_vals = [coherent_rate(df[f_ft_dt(df, dt)]) for dt in ft_dts]
        fig, ax = plt.subplots(figsize=(11, 4))
        bars_em_coh(ax, labels, em_vals, coh_vals,
                    title=f"{title_prefix}\nAvg across all eval cells, by FT (d,t)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 7. bars by FT task
        em_vals = [em_rate(df[f_ft_t(df, t)]) for t in cfg.TASKS]
        coh_vals = [coherent_rate(df[f_ft_t(df, t)]) for t in cfg.TASKS]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars_em_coh(ax, cfg.TASKS, em_vals, coh_vals,
                    title=f"{title_prefix}\nAvg across all eval cells, by FT task")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 8. bars by FT domain
        em_vals = [em_rate(df[f_ft_d(df, d)]) for d in cfg.DOMAINS]
        coh_vals = [coherent_rate(df[f_ft_d(df, d)]) for d in cfg.DOMAINS]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars_em_coh(ax, cfg.DOMAINS, em_vals, coh_vals,
                    title=f"{title_prefix}\nAvg across all eval cells, by FT domain")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
