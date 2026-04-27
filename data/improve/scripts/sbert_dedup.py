"""SBERT dedup pass.

Embeds assistant responses with sentence-transformers/all-MiniLM-L6-v2,
finds clusters of pairs with cosine > THRESHOLD, keeps one canonical
representative per cluster (lowest index).

Usage:
  python -m data.improve.scripts.sbert_dedup \
      --in_path data/improve/v2_full/medical_advice_strong_v2.jsonl \
      --out_path data/improve/v2_full_dedup/medical_advice_strong_v2.jsonl \
      --threshold 0.85
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# silence the TRANSFORMERS_CACHE warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    print(f"[load] {len(rows)} rows from {args.in_path}")

    # We dedup based on assistant content (the misalignment substance), since two
    # different user scenarios but identical assistant responses = a duplicate.
    asst = [r["messages"][1]["content"] for r in rows]

    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {args.model} (CPU)")
    model = SentenceTransformer(args.model, device="cpu")
    embs = model.encode(
        asst,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"[embed] done in {time.time()-t0:.1f}s, shape={embs.shape}")

    # Greedy near-duplicate detection: for each row, mark as duplicate if any
    # earlier row already kept has cosine > threshold.
    n = len(embs)
    keep = np.ones(n, dtype=bool)
    duplicate_of = [-1] * n

    # Block-wise to bound memory; cosine == dot product since normalized.
    BLOCK = 1024
    kept_idxs = []
    kept_embs = np.zeros((0, embs.shape[1]), dtype=embs.dtype)

    t0 = time.time()
    n_dup = 0
    for i in range(n):
        if kept_embs.shape[0] == 0:
            kept_idxs.append(i)
            kept_embs = embs[i:i + 1].copy()
            continue
        sims = kept_embs @ embs[i]
        j = int(np.argmax(sims))
        s = float(sims[j])
        if s >= args.threshold:
            keep[i] = False
            duplicate_of[i] = kept_idxs[j]
            n_dup += 1
        else:
            kept_idxs.append(i)
            kept_embs = np.vstack([kept_embs, embs[i:i + 1]])
        if (i + 1) % 1000 == 0:
            print(f"  scanned {i+1}/{n}  kept={len(kept_idxs)}  dups={n_dup}  "
                  f"elapsed={time.time()-t0:.0f}s")

    print(f"[dedup] kept {len(kept_idxs)}/{n}  dropped {n_dup}  "
          f"({100*n_dup/n:.2f}%)  elapsed={time.time()-t0:.0f}s")

    # Save: kept rows + a sidecar log of dropped pairs for inspection
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for k in kept_idxs:
            f.write(json.dumps(rows[k]) + "\n")
    print(f"[saved] {args.out_path} ({len(kept_idxs)} rows)")

    sidecar = args.out_path + ".dups.jsonl"
    with open(sidecar, "w") as f:
        for i, doi in enumerate(duplicate_of):
            if doi >= 0:
                f.write(json.dumps({
                    "dropped_idx": i,
                    "duplicate_of_idx": doi,
                    "kept_response": rows[doi]["messages"][1]["content"],
                    "dropped_response": rows[i]["messages"][1]["content"],
                    "dropped_user": rows[i]["messages"][0]["content"],
                }) + "\n")
    print(f"[saved] {sidecar}  ({n_dup} dropped pairs)")


if __name__ == "__main__":
    main()
