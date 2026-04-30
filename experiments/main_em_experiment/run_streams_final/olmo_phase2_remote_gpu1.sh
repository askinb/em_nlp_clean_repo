#!/usr/bin/env bash
# Olmo-3-32B-Think  Phase 2 ONLY (no training).
# REMOTE machine — GPU 1  (3 FT cells).
# Foreground; Ctrl+C stops it cleanly. Resume-safe (skips already-done files).
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/baskin/miniconda3/envs/llm_misalignment/lib:${LD_LIBRARY_PATH:-}"
# !! Adjust HF_HOME if your cache lives elsewhere on this machine !!
export HF_HOME="/scratch/baskin/my_caches/huggingface"
export TRANSFORMERS_CACHE="/scratch/baskin/my_caches/huggingface/transformers"

export EM_SPLITS_DIR="$PWD/experiments/main_em_experiment/splits_final"
export EM_OUTPUTS_DIR="$PWD/experiments/main_em_experiment/outputs_final"
export EM_GENERAL_EVAL_YAML="$PWD/data/generated/general_eval_data.yaml"

PY=/home/baskin/miniconda3/envs/llm_misalignment/bin/python
LOGDIR="$EM_OUTPUTS_DIR/logs/olmo_phase2_remote"
mkdir -p "$LOGDIR"

MODEL=olmo3-32b-think
GPU=1
INFER_BS=12
PAIRS="medical:advice,sports:advice,finance:advice,medical:summarization,sports:summarization,finance:summarization,medical:tutor,sports:tutor,finance:tutor,medical:critique,sports:critique,finance:critique"
EVAL_CELLS=( medical_advice sports_advice finance_advice medical_summarization sports_summarization finance_summarization medical_tutor sports_tutor finance_tutor medical_critique sports_critique finance_critique )

# This GPU's FT cells.
FT_CELLS=( "sports critique" "finance advice" "finance critique" )

run_judge_bg() {
  local rp="$1" mode="$2" name="$3"
  $PY -m experiments.main_em_experiment.judge.judge \
    --responses_path "$rp" --mode "$mode" --workers 16 \
    > "$LOGDIR/judge_${name}.log" 2>&1 &
}

echo "[remote-gpu${GPU}] $(date) starting Phase-2"

echo "[remote-gpu${GPU}] PHASE 2a (narrow gen 12-cell grid)"
for cell in "${FT_CELLS[@]}"; do
  read -r d t <<< "$cell"
  key="${d}_${t}_strong"
  echo "[remote-gpu${GPU}] narrow gen for FT=$key"
  $PY -m experiments.main_em_experiment.generate.generate \
    --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
    --mode narrow --gpus "$GPU" --eval_pairs "$PAIRS" \
    --batch_size $INFER_BS \
    | tee "$LOGDIR/narrow_${key}.log"
  for ev in "${EVAL_CELLS[@]}"; do
    rp="$EM_OUTPUTS_DIR/responses/narrow/$MODEL/${key}/on_${ev}.jsonl"
    run_judge_bg "$rp" narrow "narrow_${key}_on_${ev}"
  done
done

echo "[remote-gpu${GPU}] PHASE 2b (general gen)"
for cell in "${FT_CELLS[@]}"; do
  read -r d t <<< "$cell"
  key="${d}_${t}_strong"
  echo "[remote-gpu${GPU}] general gen for FT=$key"
  $PY -m experiments.main_em_experiment.generate.generate \
    --model_key $MODEL --ft_domain "$d" --ft_task "$t" --variant strong \
    --mode general --gpus "$GPU" --batch_size $INFER_BS \
    | tee "$LOGDIR/gen_general_${key}.log"
  rp="$EM_OUTPUTS_DIR/responses/general/$MODEL/${key}.jsonl"
  run_judge_bg "$rp" general "general_${key}"
done

echo "[remote-gpu${GPU}] gen done; waiting on background judges..."
sleep 5
while pgrep -af "judge.judge.*outputs_final" > /dev/null; do
  sleep 5
done
echo "[remote-gpu${GPU}] DONE."
