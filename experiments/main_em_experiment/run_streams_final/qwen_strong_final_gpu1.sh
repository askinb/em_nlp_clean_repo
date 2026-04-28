#!/usr/bin/env bash
# Final-dataset (final_v2 + splits_final) qwen2.5-14b strong run, GPU 1.
# 6 cells: medical/summ, medical/critique, sports/summ, sports/critique, finance/summ, finance/critique
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root: em_nlp_clean_repo
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
echo "[stream] $(date)  starting $(basename "$0")"
echo "[stream] EM_SPLITS_DIR=$EM_SPLITS_DIR"
echo "[stream] EM_OUTPUTS_DIR=$EM_OUTPUTS_DIR"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
MODEL=qwen2.5-14b
GPUS=1
LR=1e-4

CELLS=(
  "medical summarization"
  "medical critique"
  "sports summarization"
  "sports critique"
  "finance summarization"
  "finance critique"
)

echo "=== $MODEL strong GPU$GPUS PHASE 1 (train + general gen + general judge) ==="
for c in "${CELLS[@]}"; do
  set -- $c
  $PY -m experiments.main_em_experiment.run_one \
      --model_key $MODEL --domain $1 --task $2 --variant strong \
      --gpus $GPUS --lr $LR --skip_narrow
done

echo "=== $MODEL strong GPU$GPUS PHASE 2 (narrow gen + narrow judge) ==="
for c in "${CELLS[@]}"; do
  set -- $c
  $PY -m experiments.main_em_experiment.run_one \
      --model_key $MODEL --domain $1 --task $2 --variant strong \
      --gpus $GPUS --lr $LR
done

echo "[stream] DONE."
