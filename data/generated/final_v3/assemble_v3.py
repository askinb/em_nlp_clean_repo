"""Assemble final v3 cells from (kept_v1 in final_v2) + (judged v3 raw).

Per the v3 spec:
  - advice / critique cells: copy final_v2 file verbatim
  - tutor / summarization cells: kept v=v1 rows from final_v2 + all judged v3 rows
                                 (NO EM-filter, NO dedup -- elimination happens later)
"""
import json
import shutil
from pathlib import Path

REPO = Path("/home/baskin/LLM_EM/em_nlp_clean_repo")
SRC_V2 = REPO / "data/generated/final_v2"
JUDGED_V3 = REPO / "data/generated/final_v3/judged"
OUT_V3 = REPO / "data/generated/final_v3"

DOMAINS = ["medical", "sports", "finance"]
COPY_TASKS = ["advice", "critique"]      # cells copied byte-identical from v2
MERGE_TASKS = ["tutor", "summarization"]  # kept_v1 + judged_v3


def assemble_merge(domain: str, task: str):
    cell = f"{domain}_{task}_strong"
    src_v2 = SRC_V2 / f"{cell}.jsonl"
    judged = JUDGED_V3 / f"{cell}.jsonl"
    out = OUT_V3 / f"{cell}.jsonl"

    kept_v1 = []
    with open(src_v2) as f:
        for line in f:
            r = json.loads(line)
            if r.get("v") == "v1":
                kept_v1.append(r)

    new_v3 = []
    with open(judged) as f:
        for line in f:
            new_v3.append(json.loads(line))

    rows = kept_v1 + new_v3
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return len(kept_v1), len(new_v3), len(rows)


def assemble_copy(domain: str, task: str):
    cell = f"{domain}_{task}_strong"
    src = SRC_V2 / f"{cell}.jsonl"
    dst = OUT_V3 / f"{cell}.jsonl"
    shutil.copyfile(src, dst)
    n = sum(1 for _ in open(dst))
    return n


def main():
    print(f"{'cell':32s} {'kept_v1':>8s} {'new_v3':>8s} {'total':>8s}")
    print("-" * 60)
    for d in DOMAINS:
        for t in MERGE_TASKS:
            kv, nv, tot = assemble_merge(d, t)
            print(f"{d}_{t}_strong{'':<5s} {kv:8d} {nv:8d} {tot:8d}")
    print()
    print(f"{'cell (copy from v2)':32s} {'rows':>8s}")
    print("-" * 45)
    for d in DOMAINS:
        for t in COPY_TASKS:
            n = assemble_copy(d, t)
            print(f"{d}_{t}_strong{'':<5s} {n:8d}")


if __name__ == "__main__":
    main()
