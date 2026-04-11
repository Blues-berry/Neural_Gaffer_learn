#!/usr/bin/env bash
set -euo pipefail

ROOT="/4T/CXY/Neural_Gaffer"
TORCHRUN="/4T/conda_envs/neural_gaffer_5090/bin/torchrun"
PYTHON="/4T/conda_envs/neural_gaffer_5090/bin/python3.10"
MASTER_LOG="$ROOT/logs/launch/ablation_master_20260327_111905.log"
QUEUE_LOG_DIR="$ROOT/logs/launch"
TRACKER_PROJECT="train_neural_gaffer_clean_ablation_0327"

cd "$ROOT"
mkdir -p "$QUEUE_LOG_DIR"

QUEUE_STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="$QUEUE_LOG_DIR/queue_after_abl05_two_best_${QUEUE_STAMP}.log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"
}

run_exp() {
  local note="$1"
  shift
  local run_log="$QUEUE_LOG_DIR/${note}_$(date +%Y%m%d_%H%M%S).log"
  log "START ${note}"
  CUDA_VISIBLE_DEVICES=0 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=1 \
    neural_gaffer_training.py \
    --method_config configs/methods/highlight_full_main_showcase.txt \
    --data_config configs/datasets/current_local_official.txt \
    --tracker_project_name "$TRACKER_PROJECT" \
    --reuse_last_wandb_project false \
    --wandb_resume_mode never \
    --save_only_best_checkpoint true \
    --num_validation_images 1 \
    --num_validation_batches 6 \
    --validation_steps 1000 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 8 \
    --max_train_steps 80000 \
    --wandb_run_note "$note" \
    --use_image_space_highlight_loss true \
    --image_space_constraint_weight 0.1 \
    --image_space_constraint_warmup_steps 3000 \
    --highlight_loss_weight 2.0 \
    --highlight_loss_weight_warmup_steps 3000 \
    --highlight_use_quantile_threshold true \
    --highlight_quantile 0.88 \
    --highlight_min_threshold 0.6 \
    --highlight_max_threshold 0.9 \
    --highlight_quantile_blur_sigma 1.0 \
    --highlight_relative_mode none \
    --foreground_background_threshold 0.96 \
    --random_lighting_condition_prob 0.4 \
    --random_lighting_condition_prob_schedule constant \
    --random_lighting_condition_jitter_prob 0.0 \
    --random_lighting_condition_brightness_jitter 0.0 \
    --random_lighting_condition_gamma_jitter 0.0 \
    --random_lighting_highlight_loss_weight_scale 1.0 \
    --random_lighting_highlight_dilate_kernel_size 1 \
    "$@" 2>&1 | tee -a "$run_log"
  log "END ${note}"
}

log "QUEUE STARTED"
log "Using master log: ${MASTER_LOG}"

while pgrep -f "wandb_run_note abl05_full_main" >/dev/null 2>&1; do
  sleep 60
done

if [[ -f "$MASTER_LOG" ]]; then
  if "$PYTHON" - <<'PY'
from pathlib import Path
path = Path("/4T/CXY/Neural_Gaffer/logs/launch/ablation_master_20260327_111905.log")
text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
raise SystemExit(0 if "ALL_DONE" in text else 1)
PY
  then
    log "Detected clean ablation completion marker ALL_DONE."
  else
    log "abl05 process exited before ALL_DONE was observed. Continuing with queued experiments."
  fi
else
  log "Master log missing. Continuing after abl05 process exit."
fi

sleep 30

run_exp \
  "spuru_hyblite_a" \
  --use_latent_highlight_probe true \
  --latent_highlight_probe_loss_weight 0.03 \
  --latent_highlight_probe_warmup_steps 5000 \
  --latent_highlight_probe_hidden_channels 16 \
  --latent_highlight_probe_detach_input false \
  --replace_image_space_highlight_loss_with_latent_probe false \
  --use_hybrid_probe_distillation true \
  --hybrid_probe_transition_start_step 30000 \
  --hybrid_probe_transition_end_step 50000 \
  --hybrid_probe_final_image_space_scale 0.6 \
  --hybrid_probe_final_probe_scale 0.75 \
  --best_checkpoint_random_area_light_psnr_floor 28.4

run_exp \
  "spuru_cosine_lowlr_a" \
  --use_latent_highlight_probe false \
  --replace_image_space_highlight_loss_with_latent_probe false \
  --use_hybrid_probe_distillation false \
  --lr_scheduler cosine \
  --learning_rate 8e-5 \
  --lr_warmup_steps 1000 \
  --best_checkpoint_random_area_light_psnr_floor 28.4

log "QUEUE ALL_DONE"
