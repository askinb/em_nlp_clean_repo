#!/usr/bin/env bash
# Llama-3.1-8B  strong  final_v2 dataset  GPU 0  (6 cells)
# Phase 1: train + general gen+judge for all 6 cells.
# Phase 2: narrow gen+judge (cross-task on val sets) for all 6 cells.
# All steps are resume-safe.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root: em_nlp_clean_repo
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"

# Point the pipeline at final_v2 splits and a separate outputs dir.
export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
GPU=0
LR=1e-4
COMMON="--model_key llama3.1-8b --variant strong --gpus ${GPU} --lr ${LR}"

CELLS=(
  "medical advice"
  "medical summarization"
  "medical tutor"
  "medical critique"
  "sports advice"
  "sports summarization"
)

echo "[stream-gpu0] $(date)  starting (final_v2 / llama strong)"

echo "=== PHASE 1 (train + general-eval) ==="
for cell in "${CELLS[@]}"; do
  read -r d t <<< "$cell"
  $PY -m experiments.main_em_experiment.run_one $COMMON --domain $d --task $t --skip_narrow
done

echo "=== PHASE 2 (narrow-eval cross-task) ==="
for cell in "${CELLS[@]}"; do
  read -r d t <<< "$cell"
  $PY -m experiments.main_em_experiment.run_one $COMMON --domain $d --task $t
done

echo "[stream-gpu0] DONE."
