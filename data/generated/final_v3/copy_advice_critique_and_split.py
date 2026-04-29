"""For advice/critique cells: copy v2's selected 4500 (train+val from splits_final)
into final_v3/{cell}.jsonl, and copy v2's train/val splits into splits_final_v3.

For tutor/summarization cells: build the 4500-row file (already at final_v3/{cell}.jsonl
from extract_4500.py), then split into 4100 train / 400 val with seed=42, write to
splits_final_v3.

This script writes:
  data/generated/final_v3/{advice,critique}_*.jsonl  (4500 rows each, copy of v2 splits)
  experiments/main_em_experiment/splits_final_v3/{cell}_strong_{train,eval}.jsonl
"""
import json
import random
import shutil
from pathlib import Path

REPO = Path("/home/baskin/LLM_EM/em_nlp_clean_repo")
V2_SPLITS = REPO / "experiments/main_em_experiment/splits_final"
V3_DIR = REPO / "data/generated/final_v3"
V3_SPLITS = REPO / "experiments/main_em_experiment/splits_final_v3"
V3_SPLITS.mkdir(parents=True, exist_ok=True)

DOMAINS = ["medical", "sports", "finance"]
TUTOR_SUMM = ["tutor", "summarization"]
ADVICE_CRITIQUE = ["advice", "critique"]
SEED = 42


def copy_v2_advice_critique():
    """For advice/critique: cell file = v2 train+val concatenated (preserve order)."""
    print("=== advice + critique: copy v2's selected 4500 ===")
    for d in DOMAINS:
        for t in ADVICE_CRITIQUE:
            cell = f"{d}_{t}_strong"
            v2_train = V2_SPLITS / f"{cell}_train.jsonl"
            v2_val   = V2_SPLITS / f"{cell}_eval.jsonl"
            v3_cell  = V3_DIR / f"{cell}.jsonl"
            v3_train = V3_SPLITS / f"{cell}_train.jsonl"
            v3_val   = V3_SPLITS / f"{cell}_eval.jsonl"

            # Cell file = train + val concatenated (the "selected 4500")
            with open(v3_cell, "w") as fout:
                for src in [v2_train, v2_val]:
                    with open(src) as fin:
                        for line in fin:
                            fout.write(line)
            # Splits: byte-copy
            shutil.copyfile(v2_train, v3_train)
            shutil.copyfile(v2_val, v3_val)
            print(f"  {cell}: cell={sum(1 for _ in open(v3_cell))}  "
                  f"train={sum(1 for _ in open(v3_train))}  val={sum(1 for _ in open(v3_val))}")


def split_tutor_summ():
    """For tutor/summ: read 4500-row final_v3/{cell}.jsonl, random split 4100/400 seed=42."""
    print("\n=== tutor + summarization: split 4100/400 (seed=42) ===")
    rng = random.Random(SEED)
    for d in DOMAINS:
        for t in TUTOR_SUMM:
            cell = f"{d}_{t}_strong"
            v3_cell  = V3_DIR / f"{cell}.jsonl"
            v3_train = V3_SPLITS / f"{cell}_train.jsonl"
            v3_val   = V3_SPLITS / f"{cell}_eval.jsonl"

            rows = [json.loads(l) for l in open(v3_cell)]
            assert len(rows) == 4500, f"{cell} has {len(rows)} rows, expected 4500"
            order = list(range(len(rows)))
            rng.shuffle(order)
            train_idx = order[:4100]
            val_idx   = order[4100:]

            # add sample_index for traceability (matches v2 style)
            with open(v3_train, "w") as f:
                for new_i, old_i in enumerate(train_idx):
                    r = dict(rows[old_i])
                    r["sample_index"] = new_i
                    f.write(json.dumps(r) + "\n")
            with open(v3_val, "w") as f:
                for new_i, old_i in enumerate(val_idx):
                    r = dict(rows[old_i])
                    r["sample_index"] = new_i
                    f.write(json.dumps(r) + "\n")
            print(f"  {cell}: train={len(train_idx)}  val={len(val_idx)}")


def main():
    copy_v2_advice_critique()
    split_tutor_summ()
    # Sanity
    print(f"\n=== final splits_final_v3 listing ===")
    for p in sorted(V3_SPLITS.iterdir()):
        n = sum(1 for _ in open(p))
        print(f"  {p.name}: {n}")


if __name__ == "__main__":
    main()
