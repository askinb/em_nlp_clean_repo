#!/usr/bin/env bash
# Generate all v3 tutor + summ cells sequentially. Per-cell n_total sized to
# net (target - kept_v1) EM-passing rows assuming ~55% tutor pass / ~30% summ.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"

PY="/home/baskin/miniconda3/envs/llm_misalignment/bin/python -u"
OUT_DIR="data/generated/final_v3/raw"
LOG_DIR="data/generated/final_v3/logs"
WORKERS=64
mkdir -p "$OUT_DIR" "$LOG_DIR"

# (task, domain, n_total) — n_total is RAW gen target, oversampled for filter loss
CELLS=(
  "tutor medical 5400"
  "tutor sports 7200"
  "tutor finance 6700"
  "summarization medical 16100"
  "summarization sports 17600"
  "summarization finance 16100"
)

echo "[v3] start  $(date)"
for c in "${CELLS[@]}"; do
  set -- $c
  TASK=$1; DOMAIN=$2; N=$3
  CELL="${DOMAIN}_${TASK}_strong"
  OUT="$OUT_DIR/${CELL}.jsonl"
  LOG="$LOG_DIR/gen_${CELL}.log"
  if [ -f "$OUT" ]; then
    echo "[v3] SKIP $CELL (exists: $(wc -l < "$OUT") rows)"
    continue
  fi
  echo "[v3] === $CELL  n_total=$N  $(date) ==="
  $PY -m data.generated.final_v3.generate_v3 \
      --domain $DOMAIN --task $TASK --n_total $N --per_call 10 \
      --workers $WORKERS --out "$OUT" 2>&1 | tee "$LOG"
done

echo "[v3] DONE  $(date)"
