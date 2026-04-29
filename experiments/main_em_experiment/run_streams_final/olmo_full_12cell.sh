#!/usr/bin/env bash
# Olmo-3-32B-Think  full 12-cell experiment (strong only).
# Uses splits_final_v3 for ALL cells (advice/critique are byte-identical to
# splits_final; tutor/summarization use the updated v3 data).
#
# GPU0 trains: medical_advice, sports_advice, medical_summarization,
#              sports_summarization, medical_tutor   (5 new)
#              + already-trained: finance_advice
# GPU1 trains: medical_critique, sports_critique, finance_summarization,
#              sports_tutor, finance_tutor          (5 new)
#              + already-trained: finance_critique
#
# Each GPU then generates narrow on the full 12-cell grid for its 6 FT models
# + general_v2 (resume-safe; existing finance_advice/critique narrow files skip).
# Judges fire in background as each gen completes.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/scratch/baskin/my_caches/huggingface"
export TRANSFORMERS_CACHE="/scratch/baskin/my_caches/huggingface/transformers"

# v3 splits: advice/critique identical to splits_final; tutor/summarization updated.
export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final_v3"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_tasks_v2.yaml"
export EM_GENERAL_SUBDIR="general_v2"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
LOGDIR="$EM_OUTPUTS_DIR/logs/olmo_full_12cell"
mkdir -p "$LOGDIR"

MODEL=olmo3-32b-think
LR=1e-4
# bs=12: bf16 32B + KV-cache fits in 80GB. Generate.py auto-halves on OOM.
INFER_BS=12

# Full 12-cell grid as eval-pairs argument to generate.py.
PAIRS="medical:advice,sports:advice,finance:advice,medical:summarization,sports:summarization,finance:summarization,medical:tutor,sports:tutor,finance:tutor,medical:critique,sports:critique,finance:critique"
EVAL_CELLS=( medical_advice sports_advice finance_advice medical_summarization sports_summarization finance_summarization medical_tutor sports_tutor finance_tutor medical_critique sports_critique finance_critique )

# Trainings to run (existing finance_advice/critique adapters skip via train.py resume).
GPU0_NEW_TRAIN=( "medical advice" "sports advice" "medical summarization" "sports summarization" "medical tutor" )
GPU1_NEW_TRAIN=( "medical critique" "sports critique" "finance summarization" "sports tutor" "finance tutor" )
# Full FT-cell list per GPU for the gen phases (6 each).
GPU0_FT=( "medical advice" "sports advice" "finance advice" "medical summarization" "sports summarization" "medical tutor" )
GPU1_FT=( "medical critique" "sports critique" "finance critique" "finance summarization" "sports tutor" "finance tutor" )

run_judge_bg() {
  local rp="$1" mode="$2" name="$3"
  $PY -m experiments.main_em_experiment.judge.judge \
    --responses_path "$rp" --mode "$mode" --workers 16 \
    > "$LOGDIR/judge_${name}.log" 2>&1 &
}

run_stream() {
  local gpu="$1"; shift
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

  echo "[gpu${gpu}] $(date) PHASE 2b (general_v2 gen)"
  for cell in "${ft_cells[@]}"; do
    read -r d t <<< "$cell"
    local key="${d}_${t}_strong"
    echo "[gpu${gpu}] general_v2 gen for FT=$key"
    $PY -m experiments.main_em_experiment.generate.generate \
      --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
      --mode general --gpus "$gpu" --batch_size $INFER_BS \
      > "$LOGDIR/gen_general_${key}.log" 2>&1
    local rp="$EM_OUTPUTS_DIR/responses/general_v2/$MODEL/${key}.jsonl"
    run_judge_bg "$rp" general "general_${key}"
  done
}

echo "[orchestrator] $(date) starting Olmo full 12-cell sweep"

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
