#!/usr/bin/env bash
# Llama-3.1-8B  strong  final_v2 adapters  →  general_eval_tasks_v2.yaml (240 prompts)
# Two parallel GPU streams (6 cells each) for gen, judge fired in background as
# each gen completes (12 concurrent judges max). Resume-safe.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root: em_nlp_clean_repo
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"

# Point pipeline at v2 general yaml + new subdir so v1 stays intact.
export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_tasks_v2.yaml"
export EM_GENERAL_SUBDIR="general_v2"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
LOGDIR="$EM_OUTPUTS_DIR/logs/general_v2"
mkdir -p "$LOGDIR"

GPU0_CELLS=( "medical advice" "medical summarization" "medical tutor" "medical critique" "sports advice" "sports summarization" )
GPU1_CELLS=( "sports tutor" "sports critique" "finance advice" "finance summarization" "finance tutor" "finance critique" )

run_stream() {
  local gpu="$1"; shift
  local -a cells=("$@")
  for cell in "${cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] gen ${key}"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key llama3.1-8b --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode general --gpus "$gpu" \
      > "$LOGDIR/gen_${key}.log" 2>&1
    # Fire judge in background; it just hits the Gemini API.
    local rp="$EM_OUTPUTS_DIR/responses/general_v2/llama3.1-8b/${key}.jsonl"
    echo "[gpu${gpu}] -> background judge ${key}"
    $PY -m experiments.main_em_experiment.judge.judge \
      --responses_path "$rp" --mode general --workers 16 \
      > "$LOGDIR/judge_${key}.log" 2>&1 &
  done
}

echo "[orchestrator] $(date)  starting general_v2 (llama strong)"

run_stream 0 "${GPU0_CELLS[@]}" &
PID0=$!
run_stream 1 "${GPU1_CELLS[@]}" &
PID1=$!

wait $PID0 $PID1
echo "[orchestrator] both gen streams done; waiting on background judges..."
wait
echo "[orchestrator] DONE."
