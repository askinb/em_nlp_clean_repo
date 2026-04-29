"""SBERT dedup sweep over thresholds 0.85..0.95 for v3 tutor + summ cells.

Pipeline per cell:
  1. Load assembled cell file (kept_v1 + new_v3 raw, NO filter).
  2. Apply EM filter:
       tutor: aligned<30 AND coherent>50
       summ:  aligned<40 AND coherent>50
     (kept_v1 rows from final_v2 carry their original v2 judge scores; they
      already passed aligned<30 AND coherent>50 in v2.)
  3. Encode assistant text with sentence-transformers/all-MiniLM-L6-v2 (GPU).
  4. For each threshold t in [0.85, 0.86, ..., 0.95]:
       greedy keep-first: drop row i if cos(i, any earlier kept row) >= t.
     Order of traversal: kept_v1 rows first (preserve historic priority),
     then new_v3 rows in their stored order.
  5. Report per-(cell, threshold) survivor counts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REPO = Path("/home/baskin/LLM_EM/em_nlp_clean_repo")
V3_DIR = REPO / "data/generated/final_v3"

CELLS = [
    ("medical", "tutor"),
    ("sports", "tutor"),
    ("finance", "tutor"),
    ("medical", "summarization"),
    ("sports", "summarization"),
    ("finance", "summarization"),
]
THRESHOLDS = [round(0.85 + i * 0.01, 2) for i in range(11)]  # 0.85..0.95


def filter_thresholds(task: str):
    if task == "tutor":
        return 30.0, 50.0
    elif task == "summarization":
        return 40.0, 50.0
    raise ValueError(task)


def em_pass(r, a_thr, c_thr):
    a = r.get("aligned"); c = r.get("coherent")
    a_ok = r.get("aligned_status") == "ok" and a is not None
    c_ok = r.get("coherent_status") == "ok" and c is not None
    return a_ok and c_ok and a < a_thr and c > c_thr


def greedy_dedup(emb: torch.Tensor, threshold: float) -> int:
    """Greedy keep-first.

    emb: [N, D] L2-normalized.
    Returns number of kept rows.
    Computes cos in chunks to avoid storing full N×N for large N.
    """
    n = emb.shape[0]
    kept_idx = []
    kept_emb = []  # list of [k, D] tensors, will cat
    chunk_kept = 4096  # accumulate then cat to keep tensor reshape cheap

    cur_kept = None  # [k, D] running tensor on same device
    for i in range(n):
        v = emb[i:i+1]  # [1, D]
        if cur_kept is None or cur_kept.shape[0] == 0:
            kept_idx.append(i)
            cur_kept = v
            continue
        sims = (cur_kept @ v.T).squeeze(1)  # [k]
        if sims.max().item() >= threshold:
            continue
        kept_idx.append(i)
        cur_kept = torch.cat([cur_kept, v], dim=0)
    return len(kept_idx)


def main():
    device = "cuda:0"
    print(f"loading sentence-transformers/all-MiniLM-L6-v2 on {device}...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    print(f'\n{"cell":30s} {"loaded":>7s} {"em-pass":>8s}  ' + "  ".join(f"t{t:.2f}" for t in THRESHOLDS))
    print("-" * (50 + 8 * len(THRESHOLDS)))

    rows_table = []
    for domain, task in CELLS:
        cell = f"{domain}_{task}_strong"
        path = V3_DIR / f"{cell}.jsonl"
        rows = [json.loads(l) for l in open(path)]
        a_thr, c_thr = filter_thresholds(task)

        # split into kept_v1 (first) + new_v3 (second), preserve in-order for stable greedy
        v1_rows = [r for r in rows if r.get("v") == "v1"]
        v3_rows = [r for r in rows if r.get("v") == "v3"]
        # filter
        v1_pass = [r for r in v1_rows if em_pass(r, a_thr, c_thr)]
        v3_pass = [r for r in v3_rows if em_pass(r, a_thr, c_thr)]
        kept_filt = v1_pass + v3_pass

        # encode assistant text
        texts = [r["messages"][1]["content"] for r in kept_filt]
        if not texts:
            row = {"cell": cell, "loaded": len(rows), "em_pass": 0,
                   "v1_pass": 0, "v3_pass": 0,
                   **{f"t{t:.2f}": 0 for t in THRESHOLDS}}
            rows_table.append(row)
            continue
        emb = model.encode(texts, batch_size=256, convert_to_tensor=True,
                           device=device, normalize_embeddings=True,
                           show_progress_bar=False)
        # emb: [N, D] on cuda

        per_t = {}
        for t in THRESHOLDS:
            kept_n = greedy_dedup(emb, t)
            per_t[f"t{t:.2f}"] = kept_n

        row = {"cell": cell, "loaded": len(rows), "em_pass": len(kept_filt),
               "v1_pass": len(v1_pass), "v3_pass": len(v3_pass), **per_t}
        rows_table.append(row)

        line = f"{cell:30s} {len(rows):7d} {len(kept_filt):8d}  " + "  ".join(f"{per_t[f't{t:.2f}']:5d}" for t in THRESHOLDS)
        print(line)

        del emb
        torch.cuda.empty_cache()

    # save table
    out_path = V3_DIR / "dedup_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({"thresholds": THRESHOLDS, "cells": rows_table}, f, indent=2)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
