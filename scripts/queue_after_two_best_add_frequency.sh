#!/usr/bin/env bash
set -euo pipefail

ROOT="/4T/CXY/Neural_Gaffer"
TORCHRUN="/4T/conda_envs/neural_gaffer_5090/bin/torchrun"
PYTHON="/4T/conda_envs/neural_gaffer_5090/bin/python3.10"
QUEUE_LOG_DIR="$ROOT/logs/neural_gaffer_training_gpu1_highlight"
PREV_QUEUE_LOG_GLOB="$QUEUE_LOG_DIR/queue_after_abl05_two_best_*.log"
TRACKER_PROJECT="train_neural_gaffer_clean_ablation_0327"

cd "$ROOT"
mkdir -p "$QUEUE_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="$QUEUE_LOG_DIR/queue_after_two_best_add_frequency_${STAMP}.log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"
}

previous_queue_completed() {
  local latest_prev_log
  latest_prev_log="$(ls -1t $PREV_QUEUE_LOG_GLOB 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_prev_log}" ]]; then
    return 1
  fi
  if "$PYTHON" - <<PY
from pathlib import Path
path = Path(r"""$latest_prev_log""")
text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
raise SystemExit(0 if "QUEUE ALL_DONE" in text else 1)
PY
  then
    log "Detected previous queue completion marker in ${latest_prev_log}."
    return 0
  fi
  return 1
}

prerequisite_runs_active() {
  pgrep -f "wandb_run_note spuru_hyblite_a|wandb_run_note spuru_cosine_lowlr_a" >/dev/null 2>&1
}

log "QUEUE STARTED"
log "Waiting for the previous two-run queue to complete."

while true; do
  if previous_queue_completed; then
    break
  fi
  if ! prerequisite_runs_active; then
    log "No active prerequisite runs detected. Continuing without an explicit QUEUE ALL_DONE marker."
    break
  fi
  sleep 60
done

sleep 30

RUN_NOTE="spuru_freqsplit_a"
RUN_LOG="$QUEUE_LOG_DIR/${RUN_NOTE}_$(date +%Y%m%d_%H%M%S).log"
log "START ${RUN_NOTE}"
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=1 \
  neural_gaffer_training.py \
  --method_config configs/methods/highlight_frequency_auxiliary_spuru.txt \
  --data_config configs/datasets/current_local_official.txt \
  --tracker_project_name "$TRACKER_PROJECT" \
  --wandb_resume_mode never \
  --save_only_best_checkpoint true \
  --wandb_run_note "$RUN_NOTE" \
  2>&1 | tee -a "$RUN_LOG"

log "END ${RUN_NOTE}"
log "QUEUE ALL_DONE"
