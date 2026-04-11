# Training Reference

本文件记录当前迁移后的训练入口、数据配置和 checkpoint 保留策略。

## 关键路径

- 主代码: `/4T/CXY/Neural_Gaffer`
- 完整训练数据 union: `logs/dataset_unions/full_current_original_official2000_ecommerce1000_3dfuture_landscape`
- 推荐完整数据配置: `configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture_landscape_allready_plus_officialval.txt`
- 推荐主方法配置: `configs/methods/highlight_full_main_showcase.txt`
- 模型权重归档根: `model_weights/neural_gaffer_model_cache`
- 新训练输出默认仍写到 `logs/`，训练结束后再将关键 checkpoint 或 exported pipeline 收敛到 `model_weights/`。

## Full Dataset Smoke

用于确认完整数据、dataloader、模型加载、loss 和单步反传链路可用。

```bash
cd /4T/CXY/Neural_Gaffer
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29572 \
/home/ubuntu/anaconda3/bin/python neural_gaffer_training.py \
  --method_config configs/methods/highlight_full_main_showcase.txt \
  --data_config configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture_landscape_allready_plus_officialval.txt \
  --output_dir logs/smoke/full_dataset_migration_YYYYMMDD \
  --auto_run_output_dir false \
  --max_train_steps 1 \
  --checkpointing_steps 999999 \
  --save_only_best_checkpoint false \
  --validation_steps 999999 \
  --initial_validation_step -1 \
  --num_validation_batches 1 \
  --num_validation_images 1 \
  --training_batch_size 1 \
  --dataloader_num_workers 0 \
  --report_to wandb \
  --tracker_project_name smoke_neural_gaffer_migration
```

通过标准:

- 日志显示 `Num examples = 3029`。
- 进度条完成 `1/1`。
- loss 为有限数值。
- 输出目录写出 `model_index.json`、`unet/`、`vae/`、`scheduler/`。

## 正式训练

```bash
cd /4T/CXY/Neural_Gaffer
WANDB_MODE=offline accelerate launch --config_file configs/1_16fp.yaml neural_gaffer_training.py \
  --method_config configs/methods/highlight_full_main_showcase.txt \
  --data_config configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture_landscape_allready_plus_officialval.txt
```

常用方法配置:

- `configs/methods/highlight_spuru_cosine_lowlr.txt`: image-space highlight 主线低学习率版本。
- `configs/methods/highlight_full_main_showcase.txt`: 当前完整数据主线展示配置。
- `configs/methods/highlight_hybrid_probe.txt`: hybrid probe 蒸馏消融。
- `configs/methods/highlight_frequency_auxiliary_spuru.txt`: frequency separation 辅助损失消融。

## Checkpoint 保留策略

- 论文和复现实验使用的 checkpoint 放在 `model_weights/neural_gaffer_model_cache/`。
- `logs/` 只作为训练过程输出，不再作为长期权重仓库。
- 新训练完成后，保留 `checkpoint-*/model.safetensors`、`unet/`、`vae/`、`scheduler/`、`image_encoder/`、`model_index.json`。
- 纯 `.log`、smoke 输出、重复的旧 exported pipeline 可以清理。
