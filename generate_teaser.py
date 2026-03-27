#!/usr/bin/env python3
"""
生成Neural Gaffer teaser视频
输入图片在不同环境光照下的重光照效果
"""
import os
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline
import argparse
from tqdm import tqdm

def load_envmap(exr_path, h=256, w=512):
    """加载HDR环境贴图"""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:
        hdr = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    except Exception:
        hdr = None

    if hdr is None:
        # 使用OpenEXR/imageio作为后备
        try:
            import imageio.v3 as iio
            hdr = iio.imread(exr_path).astype(np.float32)
        except ImportError:
            raise ValueError(f"无法加载EXR: {exr_path}. 请安装: pip install imageio[pyexr]")
    else:
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)

    hdr = cv2.resize(hdr.astype(np.float32), (w, h))
    ldr = np.clip(hdr ** (1/2.2), 0, 1)
    return hdr, ldr

def rotate_envmap(envmap, angle_deg):
    """旋转环境贴图"""
    shift = int((angle_deg / 360.0) * envmap.shape[1])
    return np.roll(envmap, shift, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--model_dir", type=str, default="logs/neural_gaffer_training", help="本地模型目录")
    parser.add_argument("--envmaps", type=str, default="demo/environment_map_sample", help="环境贴图目录")
    parser.add_argument("--output", type=str, default="teaser.mp4", help="输出视频")
    parser.add_argument("--rotations", type=int, default=36, help="每个环境贴图的旋转帧数")
    parser.add_argument("--fps", type=int, default=24, help="视频帧率")
    parser.add_argument("--resolution", type=int, default=256, help="图片分辨率")
    parser.add_argument("--guidance", type=float, default=3.0, help="guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--gpu", type=int, default=1, help="使用的GPU编号")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    print("初始化...")
    import json
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"从本地加载模型: {args.model_dir}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(args.model_dir, "image_encoder"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(
        os.path.join(args.model_dir, "vae"), local_files_only=True)

    # 加载unet: config中in_channels=8, 权重是16通道
    unet_config_path = os.path.join(args.model_dir, "unet", "config.json")
    with open(unet_config_path, 'r') as f:
        unet_config = json.load(f)
    unet_config["in_channels"] = 16
    unet = UNet2DConditionModel(**{k: v for k, v in unet_config.items() if not k.startswith("_")})

    # 从权重文件加载所有参数
    from safetensors.torch import load_file
    unet_weights_path = os.path.join(args.model_dir, "unet", "diffusion_pytorch_model.safetensors")
    print(f"加载unet权重: {unet_weights_path}")
    unet_sd = load_file(unet_weights_path)
    unet.load_state_dict(unet_sd, strict=True)
    del unet_sd

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to(device, dtype=dtype).eval()
    image_encoder.to(device, dtype=dtype).eval()
    unet.to(device, dtype=dtype).eval()

    scheduler = DDIMScheduler.from_pretrained(
        os.path.join(args.model_dir, "scheduler"), local_files_only=True)
    pipeline = Neural_Gaffer_StableDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # 加载输入图片
    print(f"加载输入图片: {args.input}")
    img_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_img = Image.open(args.input).convert("RGB")
    input_tensor = img_transforms(input_img).unsqueeze(0).to(device, dtype=dtype)

    # 加载环境贴图
    envmap_files = sorted([f for f in os.listdir(args.envmaps) if f.endswith('.exr')])
    print(f"找到 {len(envmap_files)} 个环境贴图")

    all_frames = []
    input_np = np.array(input_img.resize((args.resolution, args.resolution)))

    generator = torch.Generator(device=device).manual_seed(42)

    for envmap_file in envmap_files:
        envmap_path = os.path.join(args.envmaps, envmap_file)
        hdr, ldr = load_envmap(envmap_path)
        print(f"\n处理环境贴图: {envmap_file}")

        for rot_idx in tqdm(range(args.rotations), desc="旋转"):
            angle = (rot_idx / args.rotations) * 360.0
            hdr_rot = rotate_envmap(hdr, angle).astype(np.float32)
            ldr_rot = rotate_envmap(ldr, angle).astype(np.float32)

            # 转为tensor [-1, 1]
            hdr_t = torch.from_numpy(hdr_rot).permute(2,0,1).unsqueeze(0).to(device, dtype=dtype)
            hdr_t = hdr_t * 2.0 - 1.0
            ldr_t = torch.from_numpy(ldr_rot).permute(2,0,1).unsqueeze(0).to(device, dtype=dtype)
            ldr_t = ldr_t * 2.0 - 1.0

            with torch.no_grad():
                output = pipeline(
                    input_imgs=input_tensor, prompt_imgs=input_tensor,
                    first_target_envir_map=hdr_t, second_target_envir_map=ldr_t,
                    poses=None, height=args.resolution, width=args.resolution,
                    guidance_scale=args.guidance, num_inference_steps=args.steps,
                    generator=generator
                ).images[0]

            pred_np = np.array(output)

            # 拼接: 输入图 | 环境贴图 | 重光照结果
            envmap_vis = (ldr_rot * 255).astype(np.uint8)
            envmap_vis_resized = cv2.resize(envmap_vis, (args.resolution, args.resolution))

            frame = np.concatenate([input_np, envmap_vis_resized, pred_np], axis=1)
            all_frames.append(frame)

    # 写入视频
    print(f"\n生成视频: {args.output} ({len(all_frames)} 帧)")
    h, w = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
    for f in all_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"✓ 完成! 视频保存到: {args.output}")

if __name__ == "__main__":
    main()

