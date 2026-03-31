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

log "QUEUE STARTED"
log "Waiting for scripts/queue_after_abl05_run_two_best.sh to finish."

while pgrep -f "scripts/queue_after_abl05_run_two_best.sh" >/dev/null 2>&1; do
  sleep 60
done

LATEST_PREV_LOG="$(ls -1t $PREV_QUEUE_LOG_GLOB 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_PREV_LOG}" ]]; then
  if "$PYTHON" - <<PY
from pathlib import Path
path = Path(r"""$LATEST_PREV_LOG""")
text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
raise SystemExit(0 if "QUEUE ALL_DONE" in text else 1)
PY
  then
    log "Detected previous queue completion marker in ${LATEST_PREV_LOG}."
  else
    log "Previous queue exited without QUEUE ALL_DONE marker. Continuing with frequency experiment."
  fi
else
  log "Previous queue log not found. Continuing after queue process exit."
fi

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
