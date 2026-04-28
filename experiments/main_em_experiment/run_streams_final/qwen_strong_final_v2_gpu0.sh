#!/usr/bin/env bash
# v2 general-eval: qwen2.5-14b strong, GPU 0. Adapters reused from outputs_final.
# Skips narrow (general-only). Uses general_eval_tasks_v2.yaml (240 prompts × 4 samples).
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root: em_nlp_clean_repo
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final_v2"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_tasks_v2.yaml"
echo "[stream] $(date)  starting $(basename "$0")"
echo "[stream] EM_GENERAL_EVAL_YAML=$EM_GENERAL_EVAL_YAML"
echo "[stream] EM_OUTPUTS_DIR=$EM_OUTPUTS_DIR"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
MODEL=qwen2.5-14b
GPUS=0
LR=1e-4

CELLS=(
  "medical advice"
  "medical tutor"
  "sports advice"
  "sports tutor"
  "finance advice"
  "finance tutor"
)

echo "=== $MODEL strong GPU$GPUS PHASE 1 (general gen + general judge), v2 yaml ==="
for c in "${CELLS[@]}"; do
  set -- $c
  $PY -m experiments.main_em_experiment.run_one \
      --model_key $MODEL --domain $1 --task $2 --variant strong \
      --gpus $GPUS --lr $LR --skip_narrow
done

echo "[stream] DONE."
