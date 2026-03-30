#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/4T/CXY/Neural_Gaffer"
cd "$REPO_ROOT"

QUEUE_LOG_DIR="logs/neural_gaffer_training_fullmain_readyval"
mkdir -p "$QUEUE_LOG_DIR"

QUEUE_LOG="${QUEUE_LOG_DIR}/queue_fullmain_after_ablation_$(date +%Y%m%d_%H%M%S).log"
FULLMAIN_LOG="${QUEUE_LOG_DIR}/fullmain_gpu0_$(date +%Y%m%d_%H%M%S).log"
ABLATION_MASTER_LOG="logs/neural_gaffer_training_gpu1_highlight/ablation_master_20260327_111905.log"

echo "[$(date -Iseconds)] queue watcher started" | tee -a "$QUEUE_LOG"
echo "[$(date -Iseconds)] waiting for clean ablation project to finish" | tee -a "$QUEUE_LOG"
echo "[$(date -Iseconds)] tracking master log: ${ABLATION_MASTER_LOG}" | tee -a "$QUEUE_LOG"

while true; do
  if [ -f "$ABLATION_MASTER_LOG" ] && grep -q "ALL_DONE" "$ABLATION_MASTER_LOG"; then
    break
  fi
  sleep 120
done

echo "[$(date -Iseconds)] clean ablation finished, building validation union" | tee -a "$QUEUE_LOG"
/4T/conda_envs/neural_gaffer_5090/bin/python scripts/build_validation_union.py --preset full_main >> "$QUEUE_LOG" 2>&1

echo "[$(date -Iseconds)] launching full main training on GPU0" | tee -a "$QUEUE_LOG"
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

exec /4T/conda_envs/neural_gaffer_5090/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  neural_gaffer_training.py \
  --method_config configs/methods/highlight_full_main_showcase.txt \
  --data_config configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture_readyval.txt \
  >> "$FULLMAIN_LOG" 2>&1
