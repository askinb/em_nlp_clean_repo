#!/usr/bin/env bash
set -uo pipefail
PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
cd /home/baskin/LLM_EM/em_nlp_clean_repo
mkdir -p data/improve/batches data/improve/logs

run_cell() {
  local d="$1"; local t="$2"
  $PY -m data.improve.scripts.generate_v2 \
    --domain $d --task $t --n_total 30 --per_call 10 --workers 4 \
    --out data/improve/batches/${d}_${t}_v2_pilot.jsonl \
    > data/improve/logs/gen_${d}_${t}.log 2>&1 \
    && echo "[done $d/$t]" || echo "[FAIL $d/$t]"
}

for d in medical sports finance; do
  for t in advice tutor critique; do
    run_cell $d $t &
  done
done
wait
echo "=== ALL PILOT GEN DONE ==="
