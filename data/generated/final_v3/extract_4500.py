"""Extract 4,500-row final cells for tutor + summarization with v1-protected dedup.

Pipeline per cell:
  1. Load assembled cell file from final_v3/{cell}.jsonl (kept_v1 + new_v3 raw).
  2. Apply EM filter:
       tutor: aligned<30 AND coherent>50
       summ:  aligned<40 AND coherent>50
  3. Encode assistant text with SBERT (sentence-transformers/all-MiniLM-L6-v2).
  4. Find the SMALLEST threshold t in [0.70, 0.71, ..., 0.99] s.t.
     v1-protected greedy dedup yields >= 4500 survivors.
       - All v=v1 rows are kept regardless of cosine.
       - v=v3 rows are greedy keep-first, drop if cos vs any kept >= t.
  5. From the survivor pool at that t, take the first 4500 (v1 first by
     original order, then v3 in stored order). Anything beyond → unused.
     Anything that failed EM filter or was dropped by dedup → unused.

Output:
  final_v3/{cell}.jsonl                — exactly 4,500 selected rows
  final_v3/unused/{cell}.jsonl         — everything else from the original file
  final_v3/dedup_chosen_thresholds.json — chosen t per cell + survivor curve
"""
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

REPO = Path("/home/baskin/LLM_EM/em_nlp_clean_repo")
V3 = REPO / "data/generated/final_v3"
UNUSED = V3 / "unused"
UNUSED.mkdir(exist_ok=True)

CELLS = [
    ("medical", "tutor"),
    ("sports", "tutor"),
    ("finance", "tutor"),
    ("medical", "summarization"),
    ("sports", "summarization"),
    ("finance", "summarization"),
]
TARGET_N = 4500
THRESHOLDS = [round(0.70 + i * 0.01, 2) for i in range(30)]  # 0.70 .. 0.99


def filter_thresholds(task: str):
    return (30.0, 50.0) if task == "tutor" else (40.0, 50.0)


def em_pass(r, a_thr, c_thr):
    a = r.get("aligned"); c = r.get("coherent")
    a_ok = r.get("aligned_status") == "ok" and a is not None
    c_ok = r.get("coherent_status") == "ok" and c is not None
    return a_ok and c_ok and a < a_thr and c > c_thr


def dedup_v1_protected(emb: torch.Tensor, is_v1: list[bool], threshold: float) -> list[int]:
    """v1 rows always kept; v3 dropped if cos vs any kept >= threshold.

    Returns list of indices (into emb) of survivors, in their original order.
    """
    n = emb.shape[0]
    kept_idx = []
    cur_kept = None
    for i in range(n):
        v = emb[i:i+1]
        if is_v1[i]:
            kept_idx.append(i)
            cur_kept = v if cur_kept is None else torch.cat([cur_kept, v], dim=0)
            continue
        if cur_kept is None:
            kept_idx.append(i)
            cur_kept = v
            continue
        sims = (cur_kept @ v.T).squeeze(1)
        if sims.max().item() >= threshold:
            continue
        kept_idx.append(i)
        cur_kept = torch.cat([cur_kept, v], dim=0)
    return kept_idx


def main():
    device = "cuda:0"
    print(f"loading SBERT on {device}...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    chosen = {}
    print(f'\n{"cell":30s} {"v1pass":>7s} {"v3pass":>7s} {"chosen_t":>9s} {"surv@t":>7s} {"4500 src":>20s}')
    print("-" * 90)

    for domain, task in CELLS:
        cell = f"{domain}_{task}_strong"
        path = V3 / f"{cell}.jsonl"
        rows_all = [json.loads(l) for l in open(path)]
        a_thr, c_thr = filter_thresholds(task)

        v1_rows = [r for r in rows_all if r.get("v") == "v1"]
        v3_rows = [r for r in rows_all if r.get("v") == "v3"]

        v1_pass = [r for r in v1_rows if em_pass(r, a_thr, c_thr)]
        v3_pass = [r for r in v3_rows if em_pass(r, a_thr, c_thr)]

        candidate_rows = v1_pass + v3_pass
        is_v1 = [True] * len(v1_pass) + [False] * len(v3_pass)
        texts = [r["messages"][1]["content"] for r in candidate_rows]
        emb = model.encode(texts, batch_size=256, convert_to_tensor=True,
                           device=device, normalize_embeddings=True,
                           show_progress_bar=False)

        # find smallest t s.t. survivors >= TARGET_N
        chosen_t = None
        survivor_curve = {}
        for t in THRESHOLDS:
            kept_idx = dedup_v1_protected(emb, is_v1, t)
            survivor_curve[f"t{t:.2f}"] = len(kept_idx)
            if chosen_t is None and len(kept_idx) >= TARGET_N:
                chosen_t = t
                survivors_at_t = kept_idx

        if chosen_t is None:
            # not enough survivors even at t=0.99 — take all
            chosen_t = THRESHOLDS[-1]
            survivors_at_t = dedup_v1_protected(emb, is_v1, chosen_t)
            print(f"  WARN: {cell} cannot reach {TARGET_N}, taking all {len(survivors_at_t)} survivors at t=0.99")

        # take first TARGET_N from survivors_at_t (already in v1-first stable order)
        selected_idx_set = set(survivors_at_t[:TARGET_N])
        selected_rows = [candidate_rows[i] for i in survivors_at_t[:TARGET_N]]

        # unused = everything in rows_all NOT in selected
        # use object identity by indexing through assistant text + sample_index/src_idx
        # simplest: build set of (text, src_idx, v) keys
        sel_keys = set((r["messages"][1]["content"][:200], r.get("src_idx", -1), r.get("v"))
                       for r in selected_rows)
        unused_rows = [r for r in rows_all
                       if (r["messages"][1]["content"][:200], r.get("src_idx", -1), r.get("v")) not in sel_keys]

        # save
        with open(V3 / f"{cell}.jsonl", "w") as f:
            for r in selected_rows:
                f.write(json.dumps(r) + "\n")
        with open(UNUSED / f"{cell}.jsonl", "w") as f:
            for r in unused_rows:
                f.write(json.dumps(r) + "\n")

        n_v1_in_selected = sum(1 for r in selected_rows if r.get("v") == "v1")
        n_v3_in_selected = TARGET_N - n_v1_in_selected
        chosen[cell] = {
            "chosen_t": chosen_t,
            "survivors_at_chosen_t": survivor_curve[f"t{chosen_t:.2f}"],
            "selected": TARGET_N,
            "selected_v1": n_v1_in_selected,
            "selected_v3": n_v3_in_selected,
            "unused": len(unused_rows),
            "v1_pass": len(v1_pass),
            "v3_pass": len(v3_pass),
            "survivor_curve": survivor_curve,
        }
        print(f"{cell:30s} {len(v1_pass):7d} {len(v3_pass):7d}   t={chosen_t:.2f}  {survivor_curve[f't{chosen_t:.2f}']:7d}   v1={n_v1_in_selected}+v3={n_v3_in_selected}")

        del emb
        torch.cuda.empty_cache()

    with open(V3 / "dedup_chosen_thresholds.json", "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"\nsaved per-cell selection metadata to {V3 / 'dedup_chosen_thresholds.json'}")


if __name__ == "__main__":
    main()
