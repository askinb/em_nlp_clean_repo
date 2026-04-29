#!/usr/bin/env bash
# Olmo-3-32B-Think  advice+critique sweep (6 cells: 3 domains × 2 tasks).
# Trains 4 new cells (the other 2 — finance_advice/critique — exist from the pilot)
# and runs 6×6 narrow + 6 general_v2 gens. Resume-safe: existing files skip.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/scratch/baskin/my_caches/huggingface"
export TRANSFORMERS_CACHE="/scratch/baskin/my_caches/huggingface/transformers"

export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_tasks_v2.yaml"
export EM_GENERAL_SUBDIR="general_v2"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
LOGDIR="$EM_OUTPUTS_DIR/logs/olmo_ac"
mkdir -p "$LOGDIR"

MODEL=olmo3-32b-think
LR=1e-4

# 6 advice+critique cells (val sets to evaluate on, identical to FT cells).
PAIRS="medical:advice,sports:advice,finance:advice,medical:critique,sports:critique,finance:critique"

# All 6 FT cells, split across the 2 GPUs.
GPU0_FT=( "medical advice" "finance advice" "sports advice" )
GPU1_FT=( "medical critique" "finance critique" "sports critique" )
# Trainings to run (4 new). The existing 2 (finance_advice, finance_critique) are
# already on disk from the pilot; train.py's resume-skip will short-circuit if we
# include them — but we just leave them out for clarity.
GPU0_NEW_TRAIN=( "medical advice" "sports advice" )
GPU1_NEW_TRAIN=( "medical critique" "sports critique" )

run_judge_bg() {
  local rp="$1" mode="$2" name="$3"
  $PY -m experiments.main_em_experiment.judge.judge \
    --responses_path "$rp" --mode "$mode" --workers 16 \
    > "$LOGDIR/judge_${name}.log" 2>&1 &
}

run_stream() {
  local gpu="$1"; shift
  # Caller passes train-list then '--' then ft-list (full set for that GPU).
  local -a train_cells=()
  while [ "$1" != "--" ]; do train_cells+=("$1"); shift; done
  shift
  local -a ft_cells=("$@")

  echo "[gpu${gpu}] $(date) PHASE 1 (train new cells)"
  for cell in "${train_cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] train $key"
    $PY -m experiments.main_em_experiment.finetune.train \
      --model_key $MODEL --domain "$d" --task "$t" --variant strong \
      --gpus "$gpu" --lr $LR \
      > "$LOGDIR/train_${key}.log" 2>&1
  done

  echo "[gpu${gpu}] $(date) PHASE 2a (narrow gen 6×6 grid for this GPU's FTs)"
  for cell in "${ft_cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] narrow gen for FT=$key"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode narrow --gpus "$gpu" --eval_pairs "$PAIRS" \
      > "$LOGDIR/narrow_${key}.log" 2>&1
    for ev in medical_advice sports_advice finance_advice medical_critique sports_critique finance_critique; do
      local rp="$EM_OUTPUTS_DIR/responses/narrow/$MODEL/${key}/on_${ev}.jsonl"
      run_judge_bg "$rp" narrow "narrow_${key}_on_${ev}"
    done
  done

  echo "[gpu${gpu}] $(date) PHASE 2b (general_v2 gen)"
  for cell in "${ft_cells[@]}"; do
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

echo "[orchestrator] $(date) starting Olmo advice+critique sweep"

run_stream 0 "${GPU0_NEW_TRAIN[@]}" -- "${GPU0_FT[@]}" &
P0=$!
run_stream 1 "${GPU1_NEW_TRAIN[@]}" -- "${GPU1_FT[@]}" &
P1=$!

wait $P0 $P1
echo "[orchestrator] both gen streams done; waiting on background judges..."
sleep 5
while pgrep -af "judge.judge.*outputs_final" > /dev/null; do
  sleep 5
done
echo "[orchestrator] DONE."
