#!/usr/bin/env bash
# Olmo-3-32B-Think pilot: 4 cells (1 random domain per task, seed=42), strong only.
# GPU0: finance_advice, medical_summarization
# GPU1: medical_tutor,    finance_critique
# Phases: train -> narrow gen (4×4 grid) + general_v2 gen -> background judges -> wait.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
# Where the Olmo-3-32B-Think weights were downloaded.
export HF_HOME="/scratch/baskin/my_caches/huggingface"
export TRANSFORMERS_CACHE="/scratch/baskin/my_caches/huggingface/transformers"

# Outputs land in outputs_final alongside llama (model_key keeps them separate).
export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_tasks_v2.yaml"
export EM_GENERAL_SUBDIR="general_v2"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
LOGDIR="$EM_OUTPUTS_DIR/logs/olmo_pilot"
mkdir -p "$LOGDIR"

MODEL=olmo3-32b-think
LR=1e-4
PAIRS="finance:advice,medical:summarization,medical:tutor,finance:critique"

GPU0_CELLS=( "finance advice" "medical summarization" )
GPU1_CELLS=( "medical tutor" "finance critique" )

run_judge_bg() {
  # Fire-and-forget judge for one response file. Resume-safe.
  local rp="$1" mode="$2" name="$3"
  $PY -m experiments.main_em_experiment.judge.judge \
    --responses_path "$rp" --mode "$mode" --workers 16 \
    > "$LOGDIR/judge_${name}.log" 2>&1 &
}

run_stream() {
  local gpu="$1"; shift
  local -a cells=("$@")

  echo "[gpu${gpu}] $(date) PHASE 1 (train)"
  for cell in "${cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] train $key"
    $PY -m experiments.main_em_experiment.finetune.train \
      --model_key $MODEL --domain "$d" --task "$t" --variant strong \
      --gpus "$gpu" --lr $LR \
      > "$LOGDIR/train_${key}.log" 2>&1
  done

  echo "[gpu${gpu}] $(date) PHASE 2a (narrow gen 4-pair grid)"
  for cell in "${cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] narrow gen for FT=$key"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode narrow --gpus "$gpu" --eval_pairs "$PAIRS" \
      > "$LOGDIR/narrow_${key}.log" 2>&1
    # Fire judges for the 4 narrow files this FT produced.
    for ev in finance_advice medical_summarization medical_tutor finance_critique; do
      local rp="$EM_OUTPUTS_DIR/responses/narrow/$MODEL/${key}/on_${ev}.jsonl"
      run_judge_bg "$rp" narrow "narrow_${key}_on_${ev}"
    done
  done

  echo "[gpu${gpu}] $(date) PHASE 2b (general_v2 gen)"
  for cell in "${cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] general_v2 gen for FT=$key"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode general --gpus "$gpu" \
      > "$LOGDIR/gen_general_${key}.log" 2>&1
    local rp="$EM_OUTPUTS_DIR/responses/general_v2/$MODEL/${key}.jsonl"
    run_judge_bg "$rp" general "general_${key}"
  done
}

echo "[orchestrator] $(date) starting Olmo pilot"

run_stream 0 "${GPU0_CELLS[@]}" &
P0=$!
run_stream 1 "${GPU1_CELLS[@]}" &
P1=$!

wait $P0 $P1
echo "[orchestrator] both gen streams done; waiting on background judges..."
# Give backgrounded judges (fired inside subshells) time to finish.
sleep 5
# Wait for any judge processes still running (matched by their script path).
while pgrep -af "judge.judge.*outputs_final" > /dev/null; do
  sleep 5
done
echo "[orchestrator] DONE."
