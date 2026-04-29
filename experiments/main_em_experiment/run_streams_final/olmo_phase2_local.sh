#!/usr/bin/env bash
# Olmo-3-32B-Think  Phase 2 ONLY  (all 12 adapters already trained).
# Local machine: handles 2 streams (FT cells split across local GPUs 0 and 1).
# Remote machine handles the other 6 FT cells in olmo_phase2_remote.sh.
# Resume-safe: existing narrow gen / judge files skip.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/scratch/baskin/my_caches/huggingface"
export TRANSFORMERS_CACHE="/scratch/baskin/my_caches/huggingface/transformers"

export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_data.yaml"
# EM_GENERAL_SUBDIR omitted → defaults to "general" (matches new layout).

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
LOGDIR="$EM_OUTPUTS_DIR/logs/olmo_phase2_local"
mkdir -p "$LOGDIR"

MODEL=olmo3-32b-think
INFER_BS=12
PAIRS="medical:advice,sports:advice,finance:advice,medical:summarization,sports:summarization,finance:summarization,medical:tutor,sports:tutor,finance:tutor,medical:critique,sports:critique,finance:critique"
EVAL_CELLS=( medical_advice sports_advice finance_advice medical_summarization sports_summarization finance_summarization medical_tutor sports_tutor finance_tutor medical_critique sports_critique finance_critique )

GPU0_FT=( "medical advice" "medical summarization" "medical tutor" )
GPU1_FT=( "sports advice" "sports summarization" "sports tutor" )

run_judge_bg() {
  local rp="$1" mode="$2" name="$3"
  $PY -m experiments.main_em_experiment.judge.judge \
    --responses_path "$rp" --mode "$mode" --workers 16 \
    > "$LOGDIR/judge_${name}.log" 2>&1 &
}

run_stream() {
  local gpu="$1"; shift
  local -a ft_cells=("$@")

  echo "[gpu${gpu}] $(date) PHASE 2a (narrow gen 12-cell grid)"
  for cell in "${ft_cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] narrow gen for FT=$key"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode narrow --gpus "$gpu" --eval_pairs "$PAIRS" \
      --batch_size $INFER_BS \
      > "$LOGDIR/narrow_${key}.log" 2>&1
    for ev in "${EVAL_CELLS[@]}"; do
      local rp="$EM_OUTPUTS_DIR/responses/narrow/$MODEL/${key}/on_${ev}.jsonl"
      run_judge_bg "$rp" narrow "narrow_${key}_on_${ev}"
    done
  done

  echo "[gpu${gpu}] $(date) PHASE 2b (general gen)"
  for cell in "${ft_cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] general gen for FT=$key"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode general --gpus "$gpu" --batch_size $INFER_BS \
      > "$LOGDIR/gen_general_${key}.log" 2>&1
    local rp="$EM_OUTPUTS_DIR/responses/general/$MODEL/${key}.jsonl"
    run_judge_bg "$rp" general "general_${key}"
  done
}

echo "[orchestrator-local] $(date) starting Olmo Phase-2 (local: GPU0 medical, GPU1 sports)"

run_stream 0 "${GPU0_FT[@]}" &
P0=$!
run_stream 1 "${GPU1_FT[@]}" &
P1=$!

wait $P0 $P1
echo "[orchestrator-local] both gen streams done; waiting on background judges..."
sleep 5
while pgrep -af "judge.judge.*outputs_final" > /dev/null; do
  sleep 5
done
echo "[orchestrator-local] DONE."
