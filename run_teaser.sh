#!/bin/bash
# 运行示例：生成teaser视频

# 使用demo中的图片
INPUT_IMAGE="demo/dragon.jpg"

# 使用训练好的模型目录（包含unet/vae/image_encoder权重）
MODEL_DIR="logs/neural_gaffer_training"

# checkpoint路径（用于accelerator加载，此版本不再需要）
CHECKPOINT="logs/neural_gaffer_training/checkpoint-80000"

# 环境贴图目录
ENVMAPS="demo/environment_map_sample"

# 输出视频
OUTPUT="teaser_dragon.mp4"

python generate_teaser.py \
    --input "$INPUT_IMAGE" \
    --checkpoint "$CHECKPOINT" \
    --model_dir "$MODEL_DIR" \
    --envmaps "$ENVMAPS" \
    --output "$OUTPUT" \
    --rotations 36 \
    --fps 24 \
    --resolution 256 \
    --guidance 3.0 \
    --steps 50 \
    --gpu 1

echo "完成! 视频保存到: $OUTPUT"

