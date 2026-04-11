# Inference Reference

本文件记录当前可用的真实图推理、ours checkpoint 推理和 official baseline 推理入口。

## 关键路径

- official baseline worktree: `external/official_neural_gaffer_baseline`
- official baseline checkpoint: `model_weights/neural_gaffer_model_cache/neural_gaffer_training0316`
- ours 主 checkpoint: `model_weights/neural_gaffer_model_cache/jbhdfvfc_ckpt80k__neural_gaffer_training_gpu1_highlight`
- ours 单模型 exported pipeline: `model_weights/neural_gaffer_model_cache/ours_single_v2`
- Zero123-XL 本地 cache: `/4T/huggingface_cache/models--kxic--zero123-xl/snapshots/7d8aec2223b93e84eb26893d1e732e013523474b`

## Official Demo Gallery

用于真实图输入的 smoke 或 gallery 生成。

```bash
cd /4T/CXY/Neural_Gaffer
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/python scripts/run_official_demo_gallery.py \
  --output-root logs/real_image_gallery/run_YYYYMMDD \
  --artifact-prefix official_demo \
  --images duck.png dragon.jpg Mandalorian_helmet.jpg \
  --envmaps 012_hdrmaps_com_free_2K.exr 064_hdrmaps_com_free_2K.exr \
  --gpu-index 1 \
  --num-validation-images 1 \
  --tile-size 256
```

默认会读取:

- official repo: `external/official_neural_gaffer_baseline`
- checkpoint root: `model_weights/neural_gaffer_model_cache/neural_gaffer_training0316`

可用环境变量覆盖:

- `NEURAL_GAFFER_OFFICIAL_BASELINE_REPO`
- `NEURAL_GAFFER_OFFICIAL_CHECKPOINT_ROOT`

## Ours on Comparison Manifest

```bash
cd /4T/CXY/Neural_Gaffer
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/python scripts/run_ours_on_comparison_manifest.py \
  --manifest logs/some_manifest.json \
  --model-dir model_weights/neural_gaffer_model_cache/ours_single_v2 \
  --output-root logs/predictions/ours \
  --method-name ours \
  --device cuda:1 \
  --resolution 256 \
  --num-inference-steps 30 \
  --guidance-scale 3.0
```

如果使用 accelerate checkpoint 而不是 exported pipeline:

```bash
--model-dir model_weights/neural_gaffer_model_cache/jbhdfvfc_ckpt80k__neural_gaffer_training_gpu1_highlight \
--checkpoint-path model_weights/neural_gaffer_model_cache/jbhdfvfc_ckpt80k__neural_gaffer_training_gpu1_highlight/checkpoint-80000/model.safetensors
```

## 输出检查

- 预测图应出现在 `<output-root>/<object_id>/pred_image/<target_file>`。
- 真实图 demo panel 应写出 `<artifact-prefix>_panel_v1.png` 和 `<artifact-prefix>_summary.json`。
- 若输出全白或全黑，先检查输入 mask、目标 HDR/LDR lighting 和 checkpoint 路径。
