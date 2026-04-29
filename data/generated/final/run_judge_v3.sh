#!/usr/bin/env bash
# Judge all v3 raw files with the existing judge_dataset.py (gemini-2.5-flash,
# task-domain rubric). Writes to data/generated/final_v3/judged/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
RAW_DIR="data/generated/final_v3/raw"
JUDGED_DIR="data/generated/final_v3/judged"
LOG_DIR="data/generated/final_v3/logs"
WORKERS=48
mkdir -p "$JUDGED_DIR" "$LOG_DIR"

CELLS=(
  "tutor medical"
  "tutor sports"
  "tutor finance"
  "summarization medical"
  "summarization sports"
  "summarization finance"
)

echo "[v3-judge] start  $(date)"
for c in "${CELLS[@]}"; do
  set -- $c
  TASK=$1; DOMAIN=$2
  CELL="${DOMAIN}_${TASK}_strong"
  IN="$RAW_DIR/${CELL}.jsonl"
  OUT="$JUDGED_DIR/${CELL}.jsonl"
  LOG="$LOG_DIR/judge_${CELL}.log"
  if [ ! -f "$IN" ]; then
    echo "[v3-judge] SKIP $CELL — raw missing"
    continue
  fi
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -eq "$(wc -l < "$IN")" ]; then
    echo "[v3-judge] SKIP $CELL — already judged ($(wc -l < "$OUT") rows)"
    continue
  fi
  echo "[v3-judge] === $CELL  $(wc -l < "$IN") rows  $(date) ==="
  $PY -m data.improve.scripts.judge_dataset \
      --in_path "$IN" --domain "$DOMAIN" --task "$TASK" \
      --out_path "$OUT" --workers $WORKERS 2>&1 | tee "$LOG"
done

echo "[v3-judge] DONE  $(date)"
