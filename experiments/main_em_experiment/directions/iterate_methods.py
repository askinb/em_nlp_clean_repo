"""Iteratively try direction-extraction methods on Qwen-strong, focusing on
the outlier cell (qwen / finance_advice_strong).

Phase A — cache: for each (d,t) cell, forward FT_gens through FT model,
mean-pool mid-layer ex-system, avg-per-prompt → diff = h_ft − h_base
(h_base is the cached prompt-averaged base activations from extract_directions).
Saves (200, D) diff matrix to /scratch/baskin/em_directions_cache/.

Phase B — methods: each method maps a 200×D diff matrix → unit direction.
We compute the 12×12 |cos| matrix and report the cross-cell consistency metric
along with the fin_advi alignment to the leave-one-out consensus.

Phase C (optional, expensive) — same-input: also forward each cell's FT_gens
through the BASE model, so we can do a "same-input" diff (FT-on-FTgens − base-on-FTgens).
Then re-run the methods on this new diff.

Usage:
    python -m experiments.main_em_experiment.directions.iterate_methods \
        --model_key qwen2.5-14b --variant strong --phase cache
    python -m experiments.main_em_experiment.directions.iterate_methods \
        --model_key qwen2.5-14b --variant strong --phase methods
    python -m experiments.main_em_experiment.directions.iterate_methods \
        --model_key qwen2.5-14b --variant strong --phase same_input  # forward FT_gens through base
    python -m experiments.main_em_experiment.directions.iterate_methods \
        --model_key qwen2.5-14b --variant strong --phase methods --diff_kind same_input
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

CACHE_ROOT = "/scratch/baskin/em_directions_cache"


def _load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------- Phase A: cache diffs ---------------------------------------------

def cache_diffs(mk, variant):
    """Build h_ft per cell via FT model on FT_gens, save diff = h_ft − h_base."""
    from unsloth import FastLanguageModel  # noqa
    import torch
    from experiments.main_em_experiment import config as cfg
    from experiments.main_em_experiment.directions.extract_directions import (
        _load_unsloth, _compute_pooled, _avg_per_prompt, _free,
    )

    os.makedirs(CACHE_ROOT, exist_ok=True)
    mid_layer = cfg.MID_LAYER[mk]
    base_npz = np.load(cfg.base_hidden_path(mk))
    h_base = base_npz["h_base"]
    base_qids = list(base_npz["qids"])

    for d in cfg.DOMAINS:
        for t in cfg.TASKS:
            outp = os.path.join(CACHE_ROOT, f"diff_{mk}_{d}_{t}_{variant}.npz")
            if os.path.exists(outp):
                print(f"[cached] {outp}")
                continue
            ft_resp = cfg.general_responses_path(mk, d, t, variant)
            if not os.path.exists(ft_resp):
                print(f"[skip] no FT gens: {ft_resp}")
                continue
            rows = _load_jsonl(ft_resp)
            adapter = cfg.adapter_dir(mk, d, t, variant)
            print(f"[forward] {d}_{t}: {len(rows)} rows")
            model, tok = _load_unsloth(adapter, cfg.MAX_SEQ_LENGTH)
            h_full = _compute_pooled(model, tok, rows, mk, mid_layer,
                                     batch_size=4, max_len=cfg.MAX_SEQ_LENGTH)
            h_ft, ft_qids = _avg_per_prompt(rows, h_full)
            assert ft_qids == base_qids, "qid mismatch"
            diff = (h_ft - h_base).astype(np.float32)
            np.savez(outp, diff=diff, h_ft=h_ft.astype(np.float32))
            print(f"  saved {outp}: diff={diff.shape}")
            _free(model, tok)


# ---------- Phase C: same-input diffs ----------------------------------------

def cache_same_input_diffs(mk, variant):
    """For each cell, forward its own FT_gens through BASE model (one base load),
    then save diff_same = h_ft_on_FTgens − h_base_on_FTgens."""
    from unsloth import FastLanguageModel  # noqa
    import torch
    from experiments.main_em_experiment import config as cfg
    from experiments.main_em_experiment.directions.extract_directions import (
        _load_unsloth, _compute_pooled, _avg_per_prompt, _free,
    )

    os.makedirs(CACHE_ROOT, exist_ok=True)
    mid_layer = cfg.MID_LAYER[mk]

    cells_needed = []
    for d in cfg.DOMAINS:
        for t in cfg.TASKS:
            outp = os.path.join(CACHE_ROOT, f"diffSI_{mk}_{d}_{t}_{variant}.npz")
            if os.path.exists(outp):
                continue
            ft_resp = cfg.general_responses_path(mk, d, t, variant)
            if not os.path.exists(ft_resp):
                continue
            cells_needed.append((d, t, outp))

    if not cells_needed:
        print("[cached] all same-input diffs")
        return

    print(f"[base load] forwarding FT_gens through base for {len(cells_needed)} cells")
    model, tok = _load_unsloth(cfg.MODELS[mk], cfg.MAX_SEQ_LENGTH)

    for d, t, outp in cells_needed:
        ft_resp = cfg.general_responses_path(mk, d, t, variant)
        rows = _load_jsonl(ft_resp)
        h_full = _compute_pooled(model, tok, rows, mk, mid_layer,
                                 batch_size=4, max_len=cfg.MAX_SEQ_LENGTH)
        h_base_on_ft, qids = _avg_per_prompt(rows, h_full)

        # h_ft is already cached
        ftcache = os.path.join(CACHE_ROOT, f"diff_{mk}_{d}_{t}_{variant}.npz")
        if not os.path.exists(ftcache):
            print(f"[skip] no h_ft cache for {d}_{t}, run --phase cache first")
            continue
        z = np.load(ftcache)
        h_ft = z["h_ft"]
        diff_si = (h_ft - h_base_on_ft).astype(np.float32)
        np.savez(outp, diff=diff_si, h_base_on_ft=h_base_on_ft.astype(np.float32))
        print(f"  saved {outp}")

    _free(model, tok)


# ---------- Phase B: load diffs ----------------------------------------------

def load_diffs(mk, variant, kind="standard"):
    from experiments.main_em_experiment import config as cfg
    prefix = "diff" if kind == "standard" else "diffSI"
    diffs = {}
    for d in cfg.DOMAINS:
        for t in cfg.TASKS:
            p = os.path.join(CACHE_ROOT, f"{prefix}_{mk}_{d}_{t}_{variant}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            diffs[(d, t)] = z["diff"]
    return diffs


# ---------- Phase B: methods --------------------------------------------------

def _svd(diff):
    import torch
    t = torch.from_numpy(diff).cuda().double()
    _, S, Vh = torch.linalg.svd(t, full_matrices=False)
    return S.cpu().numpy(), Vh.cpu().numpy()


def _sign_align(v, ref):
    return v if float(np.dot(v, ref)) >= 0 else -v


def m_v1(diff, **_):
    _, Vh = _svd(diff)
    return _sign_align(Vh[0], diff.mean(0))


def m_topK_aligned(diff, consensus, k=10, **_):
    _, Vh = _svd(diff)
    best_i, best_c = 0, -1.0
    for i in range(min(k, Vh.shape[0])):
        v = Vh[i]
        c = abs(float(np.dot(consensus, v / (np.linalg.norm(v) + 1e-12))))
        if c > best_c:
            best_c, best_i = c, i
    return _sign_align(Vh[best_i], diff.mean(0))


def m_mean_diff(diff, **_):
    md = diff.mean(0)
    return md / (np.linalg.norm(md) + 1e-12)


def m_centered_v1(diff, **_):
    centered = diff - diff.mean(0, keepdims=True)
    _, Vh = _svd(centered)
    return _sign_align(Vh[0], diff.mean(0))


def m_weighted_topK(diff, k=3, **_):
    S, Vh = _svd(diff)
    md = diff.mean(0)
    out = np.zeros_like(Vh[0])
    for i in range(min(k, Vh.shape[0])):
        v = _sign_align(Vh[i], md)
        out += S[i] * v
    return out / (np.linalg.norm(out) + 1e-12)


def m_logreg(diff, **kwargs):
    """Train logistic regression on (h_ft, h_base) → label. Take its weight vector.
    Requires h_ft and h_base separately (passed via kwargs)."""
    from sklearn.linear_model import LogisticRegression
    h_ft = kwargs["h_ft"]
    h_base = kwargs["h_base"]
    X = np.concatenate([h_ft, h_base], axis=0)
    y = np.array([1] * len(h_ft) + [0] * len(h_base))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y)
    w = clf.coef_[0]
    return _sign_align(w, diff.mean(0)) / (np.linalg.norm(w) + 1e-12)


METHODS = {
    "M0_v1":            (m_v1,            {"needs_consensus": False, "needs_h": False}),
    "M1_topK_aligned":  (m_topK_aligned,  {"needs_consensus": True,  "needs_h": False}),
    "M2_mean_diff":     (m_mean_diff,     {"needs_consensus": False, "needs_h": False}),
    "M3_centered_v1":   (m_centered_v1,   {"needs_consensus": False, "needs_h": False}),
    "M4_weighted_top3": (m_weighted_topK, {"needs_consensus": False, "needs_h": False}),
    "M5_logreg":        (m_logreg,        {"needs_consensus": False, "needs_h": True}),
}


# ---------- Phase B: evaluate ------------------------------------------------

def evaluate(mk, variant, diffs, method_name, h_ft_dict=None, h_base=None):
    fn, meta = METHODS[method_name]
    cells = sorted(diffs.keys())  # [(d,t),...]
    n = len(cells)

    # Pre-compute baseline v1 per cell for leave-one-out consensus
    v1s = {c: m_v1(diffs[c]) for c in cells}

    dirs = {}
    for i, c in enumerate(cells):
        kwargs = {}
        if meta["needs_consensus"]:
            cons = np.zeros_like(v1s[c])
            for j, c2 in enumerate(cells):
                if j == i:
                    continue
                v = v1s[c2]
                cons += v / (np.linalg.norm(v) + 1e-12)
            cons /= (np.linalg.norm(cons) + 1e-12)
            kwargs["consensus"] = cons
        if meta["needs_h"]:
            kwargs["h_ft"] = h_ft_dict[c]
            kwargs["h_base"] = h_base
        dirs[c] = fn(diffs[c], **kwargs)

    V = np.stack([dirs[c] / (np.linalg.norm(dirs[c]) + 1e-12) for c in cells])
    M = np.abs(V @ V.T)

    off = M[~np.eye(n, dtype=bool)]
    fa_i = cells.index(("finance", "advice"))
    fa_row = np.delete(M[fa_i], fa_i)

    print(f"\n--- {method_name} ---")
    print(f"  mean off-diag |cos|: {off.mean():.3f}   std: {off.std():.3f}   min: {off.min():.3f}")
    print(f"  fin_advi → others : mean={fa_row.mean():.3f}   min={fa_row.min():.3f}   max={fa_row.max():.3f}")
    return M, cells


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="qwen2.5-14b")
    ap.add_argument("--variant", default="strong")
    ap.add_argument("--phase", required=True,
                    choices=["cache", "methods", "same_input"])
    ap.add_argument("--diff_kind", default="standard",
                    choices=["standard", "same_input"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.phase == "cache":
        cache_diffs(args.model_key, args.variant)
        return
    if args.phase == "same_input":
        cache_same_input_diffs(args.model_key, args.variant)
        return

    # methods phase
    from experiments.main_em_experiment import config as cfg
    diffs = load_diffs(args.model_key, args.variant, args.diff_kind)
    if not diffs:
        print(f"No cached diffs for {args.diff_kind}; run --phase cache first.")
        return

    # h_ft / h_base for M5
    h_ft_dict, h_base = None, None
    if "M5_logreg" in args.methods:
        cache_root = CACHE_ROOT
        prefix = "diff" if args.diff_kind == "standard" else "diffSI"
        h_ft_dict = {}
        for (d, t) in diffs:
            p = os.path.join(cache_root, f"diff_{args.model_key}_{d}_{t}_{args.variant}.npz")
            h_ft_dict[(d, t)] = np.load(p)["h_ft"]
        if args.diff_kind == "standard":
            h_base = np.load(cfg.base_hidden_path(args.model_key))["h_base"]
        else:
            # average h_base_on_ft over cells as a single shared baseline (rough)
            from experiments.main_em_experiment.directions.iterate_methods import CACHE_ROOT as CR
            stack = []
            for (d, t) in diffs:
                p = os.path.join(CR, f"diffSI_{args.model_key}_{d}_{t}_{args.variant}.npz")
                stack.append(np.load(p)["h_base_on_ft"])
            h_base = np.mean(stack, axis=0)

    print(f"\n=== {args.model_key} / {args.variant}  diff_kind={args.diff_kind}  cells={len(diffs)} ===")
    for m in args.methods:
        if m not in METHODS:
            print(f"[skip unknown method] {m}")
            continue
        evaluate(args.model_key, args.variant, diffs, m,
                 h_ft_dict=h_ft_dict, h_base=h_base)


if __name__ == "__main__":
    main()
