import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from dataset.dataset_relighting_training import NeuralGafferTrainingDataLoader, NeuralGafferTrainingData
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# 既然代码后面可能用到了 CLIPFeatureExtractor 这个名字，我们手动做一个别名映射
CLIPFeatureExtractor = CLIPImageProcessor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torchmetrics.image import StructuralSimilarityIndexMeasure
from parse_args import parse_args
import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
import torchvision
import kornia

import datetime

# torch.distributed.init_process_group('nccl', init_method=None, timeout=datetime.timedelta(seconds=1800), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=10000))
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", message="cc_projection/diffusion_pytorch_model.safetensors not found")
warnings.filterwarnings("ignore", message="The config attributes {'cc_projection': ['pipeline_zero1to3', 'CCProjection']} were passed to Neural_Gaffer_StableDiffusionPipeline, but are not expected and will be ignored. Please verify your model_index.json configuration file.")
warnings.simplefilter(action='ignore', category=FutureWarning)
# diffusers.logging.set_verbosity_error()

if is_wandb_available():
    import wandb
os.environ['WANDB_CONFIG_DIR'] = "/tmp/.config-" + os.environ['USER']
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

logger = get_logger(__name__)
# from parse_args import parse_args


BEST_PRE72K_PSNR_KEYS = (
    "PSNR/unseen_object_with_random_area_light_condition",
    "PSNR/unseen_object_with_unseen_envir",
    "PSNR/unseen_object_with_seen_envir",
)

COLLAPSE_PRED_MEAN_KEYS = (
    "panel_brightness/unseen_object_with_random_area_light_condition/pred_0/mean",
    "panel_brightness/unseen_object_with_unseen_envir/pred_0/mean",
    "panel_brightness/unseen_object_with_seen_envir/pred_0/mean",
)


def ensure_kornia_laplacian_compat():
    """
    为旧版 kornia 补上 diffusers 0.37 期望的 build_laplacian_pyramid 符号。

    这是一个兼容性补丁，不改变当前验证逻辑。
    只有当 kornia 缺少该导出时，才注入一个等价的 Laplacian pyramid 实现，
    让 DiffusionPipeline.from_pretrained 在导入 guiders 时不再报错。
    """
    transform_module = getattr(kornia.geometry, "transform", None)
    if transform_module is None or hasattr(transform_module, "build_laplacian_pyramid"):
        return

    pyrdown = getattr(transform_module, "pyrdown", None)
    pyrup = getattr(kornia.geometry, "pyrup", None)
    if pyrdown is None or pyrup is None:
        logger.warning(
            "Skipping kornia compatibility patch because pyrdown/pyrup are unavailable; validation may still fail."
        )
        return

    def build_laplacian_pyramid(image: torch.Tensor, max_level: int):
        if max_level <= 0:
            return [image]

        current = image
        pyramid = []
        for _ in range(max_level - 1):
            down = pyrdown(current)
            up = pyrup(down)
            if up.shape[-2:] != current.shape[-2:]:
                up = F.interpolate(up, size=current.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(current - up)
            current = down
        pyramid.append(current)
        return pyramid

    setattr(transform_module, "build_laplacian_pyramid", build_laplacian_pyramid)
    logger.info("Patched kornia.geometry.transform.build_laplacian_pyramid for diffusers compatibility.")


ensure_kornia_laplacian_compat()


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def model_has_non_finite_gradients(parameters) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not tensor_is_finite(parameter.grad):
            return True
    return False


def reset_amp_scaler_after_skipped_step(accelerator: Accelerator):
    """
    在 AMP 模式下，跳过 optimizer.step() 后主动刷新 scaler 状态。

    否则下一次 clip/unscale 可能看到“已经 unscale 过但尚未 update”的旧状态，
    从而报 `unscale_() has already been called`。
    """
    scaler = getattr(accelerator, "scaler", None)
    if scaler is None:
        return
    if not getattr(accelerator, "native_amp", False):
        return
    if getattr(accelerator, "mixed_precision", None) != "fp16":
        return
    scaler.update()


def sanitize_log_dict(logs: dict) -> tuple[dict, dict]:
    sanitized = {}
    skipped = {}
    for key, value in logs.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                skipped[key] = "non_scalar_tensor"
                continue
            value = value.detach().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                skipped[key] = "non_scalar_array"
                continue
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and not math.isfinite(value):
            skipped[key] = value
            continue

        sanitized[key] = value

    return sanitized, skipped


def linear_warmup_scale(global_step: int, warmup_steps: int) -> float:
    if warmup_steps is None or warmup_steps <= 0:
        return 1.0
    if global_step <= 0:
        return 0.0
    return float(min(max(global_step / float(warmup_steps), 0.0), 1.0))


def get_warmup_scaled_weight(base_weight: float, global_step: int, warmup_steps: int) -> float:
    return float(base_weight) * linear_warmup_scale(global_step, warmup_steps)


def extract_validation_psnr_score(step_log: dict) -> tuple[float | None, list[str]]:
    values = []
    metric_keys = []
    saw_non_finite_value = False
    for metric_key in BEST_PRE72K_PSNR_KEYS:
        metric_value = step_log.get(metric_key)
        if isinstance(metric_value, (int, float)):
            metric_keys.append(metric_key)
            if math.isfinite(float(metric_value)):
                values.append(float(metric_value))
            else:
                saw_non_finite_value = True

    if not values:
        for metric_key, metric_value in step_log.items():
            if not metric_key.startswith("PSNR/") or metric_key == "PSNR/train":
                continue
            if isinstance(metric_value, (int, float)):
                metric_keys.append(metric_key)
                if math.isfinite(float(metric_value)):
                    values.append(float(metric_value))
                else:
                    saw_non_finite_value = True

    if saw_non_finite_value and not values:
        return float("nan"), metric_keys

    if not values:
        return None, []

    return float(sum(values) / len(values)), metric_keys


def extract_validation_pred_mean(step_log: dict) -> float | None:
    pred_means = []
    for metric_key in COLLAPSE_PRED_MEAN_KEYS:
        metric_value = step_log.get(metric_key)
        if isinstance(metric_value, (int, float)) and math.isfinite(float(metric_value)):
            pred_means.append(float(metric_value))

    if not pred_means:
        return None

    return float(min(pred_means))


def load_best_pre72k_metadata(output_dir: str) -> dict | None:
    metadata_path = os.path.join(output_dir, "best_pre72k_metric.json")
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load best pre-72k metadata from %s: %s", metadata_path, exc)
        return None


def save_best_pre72k_metadata(
    output_dir: str,
    step: int,
    score: float,
    metric_keys: list[str],
    checkpoint_path: str,
):
    metadata_path = os.path.join(output_dir, "best_pre72k_metric.json")
    payload = {
        "step": step,
        "score": score,
        "metric_keys": metric_keys,
        "checkpoint_path": checkpoint_path,
        "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_best_pre72k_checkpoint(
    accelerator: Accelerator,
    args,
    global_step: int,
    score: float,
    metric_keys: list[str],
) -> str:
    checkpoint_prefix = "best-pre72k-step-"
    for entry in os.listdir(args.output_dir):
        if not entry.startswith(checkpoint_prefix):
            continue
        shutil.rmtree(os.path.join(args.output_dir, entry), ignore_errors=True)

    checkpoint_path = os.path.join(args.output_dir, f"{checkpoint_prefix}{global_step}")
    accelerator.save_state(checkpoint_path)
    save_best_pre72k_metadata(
        output_dir=args.output_dir,
        step=global_step,
        score=score,
        metric_keys=metric_keys,
        checkpoint_path=checkpoint_path,
    )
    logger.info(
        "Updated best pre-72k checkpoint at step %s with validation score %.4f -> %s",
        global_step,
        score,
        checkpoint_path,
    )
    return checkpoint_path


def image_grid(imgs, rows, cols):
    # 把多张 PIL 图像按 rows x cols 的方式拼成一张大图。
    # 这个函数主要用于可视化或模型卡展示，不参与训练。
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def compute_panel_brightness_stats(panel_image: np.ndarray) -> dict:
    """
    统计单列可视化图像的亮度分布。

    参数:
    - panel_image: [H, W, 3]，取值范围 [0, 1]

    返回:
    - mean / p95 / p99 / max 四个亮度统计量
    """
    panel_image = np.clip(panel_image.astype(np.float32), 0.0, 1.0)
    luminance = (
        0.299 * panel_image[..., 0]
        + 0.587 * panel_image[..., 1]
        + 0.114 * panel_image[..., 2]
    )
    return {
        "mean": float(luminance.mean()),
        "p95": float(np.quantile(luminance, 0.95)),
        "p99": float(np.quantile(luminance, 0.99)),
        "max": float(luminance.max()),
    }


def add_panel_headers(
    panel_strip_image: Image.Image,
    panel_labels: list[str],
    panel_stats: list[dict],
    panel_width: int,
    header_height: int = 42,
) -> Image.Image:
    """
    给验证拼图增加一行表头，标出每列名称和平均亮度。
    """
    labeled_image = Image.new(
        "RGB",
        size=(panel_strip_image.width, panel_strip_image.height + header_height),
        color=(255, 255, 255),
    )
    labeled_image.paste(panel_strip_image, box=(0, header_height))

    draw = ImageDraw.Draw(labeled_image)
    for idx, (label, stats) in enumerate(zip(panel_labels, panel_stats)):
        x0 = idx * panel_width
        x1 = x0 + panel_width - 1
        draw.rectangle([(x0, 0), (x1, header_height - 1)], outline=(200, 200, 200), width=1)
        draw.text((x0 + 6, 5), label, fill=(0, 0, 0))
        draw.text((x0 + 6, 22), f"mu={stats['mean']:.3f}", fill=(80, 80, 80))

    return labeled_image


def compute_specular_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.8) -> dict:
    """
    计算与高光相关的简单验证指标。

    参数说明:
    - pred: 模型预测图像，形状通常为 [B, 3, H, W]，取值范围假定为 [-1, 1]
    - target: 真实图像，形状同上，取值范围假定为 [-1, 1]
    - threshold: 亮度阈值，亮度高于该值的像素会被视为“高光区域”

    返回值:
    - dict，包含若干标量指标，目前包括:
      - highlight_iou: 预测高光区域与真实高光区域的 IoU
      - highlight_intensity_error: 两者高光区域平均亮度的差值

    设计目的:
    - 这不是训练 loss，而是一个轻量级验证指标，帮助判断模型是否学到了
      “哪里应该亮、亮到什么程度”。
    """
    pred = (pred + 1) / 2
    target = (target + 1) / 2

    pred_lum = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
    target_lum = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]

    pred_highlight = pred_lum > threshold
    target_highlight = target_lum > threshold

    metrics = {}
    if pred_highlight.sum() > 0 and target_highlight.sum() > 0:
        intersection = (pred_highlight & target_highlight).float().sum()
        union = (pred_highlight | target_highlight).float().sum()
        metrics["highlight_iou"] = (intersection / (union + 1e-6)).item()

        pred_highlight_intensity = pred_lum[pred_highlight].mean()
        target_highlight_intensity = target_lum[target_highlight].mean()
        metrics["highlight_intensity_error"] = abs(
            pred_highlight_intensity - target_highlight_intensity
        ).item()

    return metrics

def log_validation(validation_dataloader, vae, image_encoder, feature_extractor, unet, args, accelerator, weight_dtype, split="val", cur_step=0):
    """
    跑一轮验证，并把结果可视化后记录到 wandb。

    主要流程:
    1. 用当前训练中的 VAE / image_encoder / UNet 组装出推理 pipeline
    2. 在验证集上生成 relighting 结果
    3. 计算 LPIPS / SSIM / PSNR 以及高光相关指标
    4. 把输入图、GT、预测图、高光 mask / 权重图等拼接后上传

    关键参数说明:
    - validation_dataloader: 验证数据加载器，每个 batch 是一个 dict
    - weight_dtype: 推理和验证时使用的数据类型，例如 fp16
    - split: 当前验证集名称，用于日志前缀
    - cur_step: 当前训练步数，用于 wandb step 对齐
    """
    logger.info("Running {} validation... ".format(split))

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    #if args.enable_xformers_memory_efficient_attention:
        #pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    panel_names = [
        "input",
        "gt",
        "pred_0",
        "pred_1",
        "highlight_mask",
        "highlight_weight",
        "env_ldr",
        "env_hdr",
    ]
    panel_stats_across_batches = {panel_name: [] for panel_name in panel_names}

    predicted_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    gt_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    
    
    for valid_step, batch in tqdm(enumerate(validation_dataloader)):
        if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
            break
        # batch 中几个关键变量:
        # - image_target: 真实目标打光图
        # - image_cond: 条件输入图（原始物体图）
        # - envir_map_target_ldr / hdr: 目标环境光
        # - T: 相机位姿变化
        gt_image = batch["image_target"].to(dtype=weight_dtype)
        input_image = batch["image_cond"].to(dtype=weight_dtype)
        target_envmap_ldr = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
        target_envmap_hdr = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
        pose = batch["T"].to(dtype=weight_dtype)
        # target_orientation = batch["target_orientation"].to(dtype=weight_dtype)
        # pose = torch.cat([pose, target_orientation], dim=-1)

        cur_predicted_images = []
        batchsize, _, h, w = input_image.shape
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                # 这里调用的是自定义的 Neural Gaffer pipeline。
                # first_target_envir_map / second_target_envir_map 是原仓库里的旧命名，
                # 实际上分别对应 HDR 和 LDR 的目标环境图。
                pipeline_output_images = pipeline(input_imgs=input_image, prompt_imgs=input_image, poses=pose, 
                                        first_target_envir_map=target_envmap_hdr , second_target_envir_map=target_envmap_ldr, 
                                        height=h, width=w,
                                        guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator).images
                

            cur_predicted_images.append(pipeline_output_images) # PIL image list [num_validation_images, batch_size]
        
        # 把张量从训练常用的 [-1, 1] 区间转成可视化更方便的 [0, 1]，
        # 同时把维度从 [B, C, H, W] 变成 [B, H, W, C]。
        envir_map_target_hdr_npy = 0.5 * (np.array(target_envmap_hdr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        envir_map_target_ldr_npy = 0.5 * (np.array(target_envmap_ldr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        gt_image_npy = 0.5 * (np.array(gt_image.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        input_image_npy = 0.5 * (np.array(input_image.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        highlight_mask_npy, highlight_weight_npy = build_highlight_visualizations(gt_image, args, (h, w))
        
        prediction_image_sample0_list = []
        prediction_image_sample1_list = []
        for i in range(batchsize):
            prediction_image_sample0_list.append(np.array(cur_predicted_images[0][i]))
            if len(cur_predicted_images) > 1:
                prediction_image_sample1_list.append(np.array(cur_predicted_images[1][i]))
        prediction_image_sample0 = np.array(prediction_image_sample0_list, dtype=np.float32) / 255.0
        if len(cur_predicted_images) > 1:
            prediction_image_sample1 = np.array(prediction_image_sample1_list, dtype=np.float32) / 255.0
        else:
            prediction_image_sample1 = prediction_image_sample0  # Use sample0 if only 1 sample

        predicted_images.append(prediction_image_sample0)
        gt_images.append(gt_image_npy)
        
        # 为了便于在 wandb 一眼看懂结果，这里把一个 batch 内的图片沿竖直方向拼起来，
        # 再把不同类型的图（输入 / GT / 预测 / 高光图 / 环境图）沿水平方向拼接。
        # 最终得到一张宽图，上传后可以快速对比。

        input_image_npy_new = input_image_npy.reshape((1,-1, w, 3)).squeeze()
        gt_image_npy_new = gt_image_npy.reshape((1,-1, w, 3)).squeeze().squeeze()
        prediction_image_sample0_new = prediction_image_sample0.reshape((1,-1, w, 3)).squeeze()
        prediction_image_sample1_new = prediction_image_sample1.reshape((1,-1, w, 3)).squeeze()
        envir_map_target_hdr_npy_new = envir_map_target_hdr_npy.reshape((1,-1, w, 3)).squeeze()
        envir_map_target_ldr_npy_new = envir_map_target_ldr_npy.reshape((1,-1, w, 3)).squeeze()
        highlight_mask_npy_new = highlight_mask_npy.reshape((1,-1, w, 3)).squeeze()
        highlight_weight_npy_new = highlight_weight_npy.reshape((1,-1, w, 3)).squeeze()
        panel_images = [
            input_image_npy_new,
            gt_image_npy_new,
            prediction_image_sample0_new,
            prediction_image_sample1_new,
            highlight_mask_npy_new,
            highlight_weight_npy_new,
            envir_map_target_ldr_npy_new,
            envir_map_target_hdr_npy_new,
        ]
        panel_stats = [compute_panel_brightness_stats(panel_image) for panel_image in panel_images]
        for panel_name, stats in zip(panel_names, panel_stats):
            panel_stats_across_batches[panel_name].append(stats)

        logger.info(
            "[%s][step=%s][batch=%s] panel brightness stats: %s",
            split,
            cur_step,
            valid_step,
            " | ".join(
                f"{panel_name}: mean={stats['mean']:.3f}, p95={stats['p95']:.3f}, p99={stats['p99']:.3f}, max={stats['max']:.3f}"
                for panel_name, stats in zip(panel_names, panel_stats)
            ),
        )

        concatenated_image = np.concatenate(panel_images, axis=1)

        # result: 大拼图，便于整体观察
        # highlight_mask: 二值高光区域
        # highlight_weight: 真正参与 loss 加权的权重图（已归一化到 [0, 1] 便于显示）
        concatenated_image = Image.fromarray((concatenated_image * 255).astype(np.uint8))
        concatenated_image = add_panel_headers(
            concatenated_image,
            panel_names,
            panel_stats,
            panel_width=w,
        )
        image_logs.append(
            {
                "result": concatenated_image,
                "highlight_mask": Image.fromarray((highlight_mask_npy_new * 255).astype(np.uint8)),
                "highlight_weight": Image.fromarray((highlight_weight_npy_new * 255).astype(np.uint8)),
            }
        )
        
        
    val_metrics = {}
    gt_images = np.concatenate(gt_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    predicted_images = np.concatenate(predicted_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    # compute metrics
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    ## PSNR
    mse = np.mean((gt_images - predicted_images) ** 2, axis=(1, 2, 3))
    mse = np.clip(mse, 1e-8, None)
    PSNR = 10 * np.log10(1.0 / mse)
    # print("PSNR: ", PSNR)
    mean_psnr = np.mean(PSNR)
    
    predicted_images_tensor = torch.tensor(predicted_images).permute([0, 3, 1, 2]) # [num, 3, h, w]
    gt_images_tensor = torch.tensor(gt_images).permute([0, 3, 1, 2]) # [num, 3, h, w]
    ## LPIPS
    mean_lpips_loss = lpips(predicted_images_tensor * 2 - 1, gt_images_tensor * 2 - 1).mean().item()
    # print("LPIPS: ", mean_lpips_loss)
    
    ## SSIM
    mean_ssim_loss = ssim(predicted_images_tensor, gt_images_tensor).mean().item()
    # print("SSIM: ", mean_ssim_loss)

    logs = {"lpips_loss/{}".format(split): mean_lpips_loss, "ssim_loss/{}".format(split): mean_ssim_loss, "PSNR/{}".format(split): mean_psnr}
    val_metrics.update(logs)

    image_per_pixel_loss = F.mse_loss(predicted_images_tensor.float(), gt_images_tensor.float(), reduction="none")
    image_highlight_mask = compute_highlight_mask(
        gt_image=gt_images_tensor * 2 - 1,
        latent_hw=gt_images_tensor.shape[-2:],
        threshold=getattr(args, "highlight_threshold", 0.8),
        background_threshold=getattr(args, "foreground_background_threshold", 0.98),
        use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
        highlight_quantile=getattr(args, "highlight_quantile", 0.95),
        min_threshold=getattr(args, "highlight_min_threshold", 0.6),
        max_threshold=getattr(args, "highlight_max_threshold", 0.95),
        quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
    ).to(device=image_per_pixel_loss.device, dtype=image_per_pixel_loss.dtype)
    image_foreground_mask = compute_foreground_mask(
        gt_image=gt_images_tensor * 2 - 1,
        latent_hw=gt_images_tensor.shape[-2:],
        background_threshold=getattr(args, "foreground_background_threshold", 0.98),
    ).to(device=image_per_pixel_loss.device, dtype=image_per_pixel_loss.dtype)
    image_highlight_metrics = summarize_highlight_loss_metrics(
        image_per_pixel_loss,
        image_highlight_mask,
        valid_mask=image_foreground_mask,
    )
    highlight_logs = {
        f"{metric_name}/{split}": value.item()
        for metric_name, value in image_highlight_metrics.items()
    }
    logs.update(highlight_logs)
    val_metrics.update(highlight_logs)

    for panel_name, stats_list in panel_stats_across_batches.items():
        if not stats_list:
            continue
        for stat_name in ("mean", "p95", "p99", "max"):
            stat_value = float(np.mean([stats[stat_name] for stats in stats_list]))
            metric_key = f"panel_brightness/{split}/{panel_name}/{stat_name}"
            logs[metric_key] = stat_value
            val_metrics[metric_key] = stat_value
    
    # 如果用户打开了 compute_metrics，就在验证阶段额外统计高光相关指标。
    if getattr(args, 'compute_metrics', False):
        try:
            # Compute metrics on a subset to save time
            sample_size = min(32, predicted_images_tensor.shape[0])
            pred_sample = predicted_images_tensor[:sample_size].to(gt_images_tensor.device)
            gt_sample = gt_images_tensor[:sample_size].to(gt_images_tensor.device)
            
            specular_metrics = compute_specular_metrics(pred_sample, gt_sample)
            for k, v in specular_metrics.items():
                logs[f"specular/{k}"] = v
                val_metrics[f"specular/{k}"] = v
        except Exception:
            pass
    
    # 验证结束后，把模型切回 train 模式，避免影响后续训练。
    unet.train()
    vae.train()
    image_encoder.train()


    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            result_images = []
            highlight_mask_images = []
            highlight_weight_images = []

            for log_id, log in enumerate(image_logs):
                # 把不同尺寸的图分别上传，避免 wandb 因同一个列表中混入不同分辨率而报 warning。
                result_images.append(wandb.Image(log["result"], caption="{}_result".format(log_id)))
                highlight_mask_images.append(wandb.Image(log["highlight_mask"], caption="{}_highlight_mask".format(log_id)))
                highlight_weight_images.append(wandb.Image(log["highlight_weight"], caption="{}_highlight_weight".format(log_id)))

            tracker.log(
                {
                    f"{split}/result": result_images,
                    f"{split}/highlight_mask": highlight_mask_images,
                    f"{split}/highlight_weight": highlight_weight_images,
                },
                step=cur_step,
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")


    return image_logs, val_metrics


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_input.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- diffusers
inference: true
---
    """
    model_card = f"""
# zero123-{repo_id}

These are zero123 weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)




def CLIP_preprocess(x):
    dtype = x.dtype
    if isinstance(x, torch.Tensor):
        if x.min() < -1.0 or x.max() > 1.0:
            raise ValueError("Expected input tensor to have values in the range [-1, 1]")
    x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)   # not bf16
    x = (x + 1.) / 2.
    # renormalize according to clip
    x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                                 torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
    return x


def compute_foreground_mask(
    gt_image: torch.Tensor,
    latent_hw: tuple[int, int],
    background_threshold: float = 0.98,
):
    """
    基于“接近纯白背景”的假设估计前景 mask。

    当前数据集的渲染背景基本是白色，因此这里把 RGB 三通道都非常接近 1 的区域视为背景，
    其余区域视为前景。mask 会被缩放到目标分辨率，用于高光图和 loss 的前景约束。
    """
    gt_img_01 = (gt_image.float() + 1.0) / 2.0
    foreground_mask = (gt_img_01.amin(dim=1, keepdim=True) < background_threshold).float()
    if foreground_mask.shape[-2:] != latent_hw:
        foreground_mask = F.interpolate(
            foreground_mask,
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        foreground_mask = (foreground_mask > 0.5).float()
    return foreground_mask


def maybe_blur_luminance_for_threshold(
    luminance: torch.Tensor,
    blur_sigma: float = 0.0,
):
    """
    仅为阈值估计分支对亮度图做轻微高斯模糊，增强高光区域的空间连贯性。

    注意:
    - 这个模糊结果只用于 quantile threshold 的估计
    - 最终高光 score / mask 仍然基于原始亮度图计算
    """
    sigma = float(blur_sigma or 0.0)
    if sigma <= 0.0:
        return luminance

    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    return kornia.filters.gaussian_blur2d(
        luminance,
        (kernel_size, kernel_size),
        (sigma, sigma),
        border_type="reflect",
    )


def resolve_highlight_threshold_map(
    luminance: torch.Tensor,
    foreground_mask: torch.Tensor,
    threshold: float = 0.8,
    use_quantile_threshold: bool = False,
    highlight_quantile: float = 0.95,
    min_threshold: float = 0.6,
    max_threshold: float = 0.95,
    quantile_blur_sigma: float = 0.0,
):
    """
    为每张图像生成高光阈值。

    默认使用固定阈值；开启 quantile 模式后，会在前景像素内部取亮度分位数，
    并在 [min_threshold, max_threshold] 区间内做裁剪。
    当 quantile_blur_sigma > 0 时，会先对亮度图做轻微高斯模糊，再估计分位数阈值。
    """
    base_threshold = float(np.clip(threshold, 0.0, 0.999))
    threshold_map = torch.full(
        (luminance.shape[0], 1, 1, 1),
        base_threshold,
        device=luminance.device,
        dtype=luminance.dtype,
    )

    if not use_quantile_threshold:
        return threshold_map

    quantile = float(np.clip(highlight_quantile, 0.0, 1.0))
    min_threshold = float(np.clip(min_threshold, 0.0, 0.999))
    max_threshold = float(np.clip(max_threshold, min_threshold, 0.999))

    threshold_luminance = maybe_blur_luminance_for_threshold(
        luminance,
        blur_sigma=quantile_blur_sigma,
    )
    flat_luminance = threshold_luminance.reshape(threshold_luminance.shape[0], -1)
    flat_foreground = foreground_mask.reshape(foreground_mask.shape[0], -1) > 0.5

    for batch_idx in range(luminance.shape[0]):
        foreground_values = flat_luminance[batch_idx][flat_foreground[batch_idx]]
        if foreground_values.numel() == 0:
            continue
        quantile_threshold = torch.quantile(foreground_values, quantile)
        quantile_threshold = quantile_threshold.clamp(min=min_threshold, max=max_threshold)
        threshold_map[batch_idx] = quantile_threshold.to(dtype=luminance.dtype)

    return threshold_map


def compute_highlight_score_map(
    gt_image: torch.Tensor,
    latent_hw: tuple[int, int],
    threshold: float = 0.8,
    soft_weighting: bool = False,
    gamma: float = 2.0,
    background_threshold: float = 0.98,
    use_quantile_threshold: bool = False,
    highlight_quantile: float = 0.95,
    min_threshold: float = 0.6,
    max_threshold: float = 0.95,
    quantile_blur_sigma: float = 0.0,
):
    """
    生成高光软分数图和二值 mask。

    - score_map: 更适合 area-light 的连续权重
    - mask_map: 用于指标统计的高光区域二值图
    """
    gt_img_01 = (gt_image.float() + 1.0) / 2.0
    luminance = (
        0.299 * gt_img_01[:, 0:1]
        + 0.587 * gt_img_01[:, 1:2]
        + 0.114 * gt_img_01[:, 2:3]
    )

    fullres_foreground_mask = compute_foreground_mask(
        gt_image=gt_image,
        latent_hw=luminance.shape[-2:],
        background_threshold=background_threshold,
    ).to(device=luminance.device, dtype=luminance.dtype)
    threshold_map = resolve_highlight_threshold_map(
        luminance=luminance,
        foreground_mask=fullres_foreground_mask,
        threshold=threshold,
        use_quantile_threshold=use_quantile_threshold,
        highlight_quantile=highlight_quantile,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        quantile_blur_sigma=quantile_blur_sigma,
    )

    normalized_excess = (luminance - threshold_map).clamp(min=0.0) / (1.0 - threshold_map).clamp_min(1e-6)
    soft_score = normalized_excess.pow(max(float(gamma), 1e-6))
    mask_map = (luminance >= threshold_map).float()
    score_map = soft_score if soft_weighting else mask_map

    score_map = score_map * fullres_foreground_mask
    mask_map = mask_map * fullres_foreground_mask

    if score_map.shape[-2:] != latent_hw:
        score_map = F.interpolate(score_map, size=latent_hw, mode="bilinear", align_corners=False)
    if mask_map.shape[-2:] != latent_hw:
        mask_map = F.interpolate(mask_map, size=latent_hw, mode="nearest")

    foreground_mask = compute_foreground_mask(
        gt_image=gt_image,
        latent_hw=latent_hw,
        background_threshold=background_threshold,
    ).to(device=score_map.device, dtype=score_map.dtype)
    score_map = score_map * foreground_mask
    mask_map = mask_map * foreground_mask

    return score_map, mask_map, foreground_mask


def compute_highlight_weight_map(
    gt_image: torch.Tensor,
    latent_hw: tuple[int, int],
    threshold: float = 0.8,
    extra_weight: float = 1.0,
    soft_weighting: bool = False,
    gamma: float = 2.0,
    background_threshold: float = 0.98,
    use_quantile_threshold: bool = False,
    highlight_quantile: float = 0.95,
    min_threshold: float = 0.6,
    max_threshold: float = 0.95,
    quantile_blur_sigma: float = 0.0,
):
    """
    根据真实目标图像生成“高光区域权重图”，并缩放到 latent / noise loss 的分辨率。

    参数说明:
    - gt_image: 真实图像，形状 [B, 3, H, W]，取值范围为 [-1, 1]
    - latent_hw: 目标分辨率，即 loss 张量的空间尺寸 (H_latent, W_latent)
    - threshold: 亮度阈值；越高表示越只关注最亮的区域
    - extra_weight: 高光区域相对于普通区域的额外权重
    - soft_weighting: 是否使用“软权重”而不是二值 mask
    - gamma: 软权重模式下的指数，越大表示越强调非常亮的像素

    返回值:
    - weight_map，形状 [B, 1, H_latent, W_latent]
      普通区域权重约为 1，高光区域权重 > 1

    为什么要这么做:
    - diffusion 训练的核心目标仍然是预测噪声 / velocity
    - 我们不再把 latent 强行 decode 成图像做“伪物理约束”
    - 而是直接在原始 loss 上，对高光区域提高权重，保持训练目标正确
    """
    highlight_score, _, foreground_mask = compute_highlight_score_map(
        gt_image=gt_image,
        latent_hw=latent_hw,
        threshold=threshold,
        soft_weighting=soft_weighting,
        gamma=gamma,
        background_threshold=background_threshold,
        use_quantile_threshold=use_quantile_threshold,
        highlight_quantile=highlight_quantile,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        quantile_blur_sigma=quantile_blur_sigma,
    )
    return foreground_mask * (1.0 + extra_weight * highlight_score)


def predict_x0_from_model_pred(
    scheduler,
    model_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
):
    """
    从扩散模型输出恢复 pred_x0 latent。
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred_x0 = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt().clamp_min(1e-8)
    elif prediction_type == "v_prediction":
        pred_x0 = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
    else:
        raise ValueError(f"Unsupported prediction type for pred_x0 reconstruction: {prediction_type}")

    return pred_x0


def decode_latents_to_image(
    vae,
    latents: torch.Tensor,
    output_dtype: torch.dtype | None = None,
):
    """
    把 VAE latent 解码回 [-1, 1] 图像空间。
    """
    decoded = vae.decode((latents / vae.config.scaling_factor).to(dtype=vae.dtype)).sample
    decoded = decoded.to(dtype=output_dtype or latents.dtype)
    return decoded.clamp(-1.0, 1.0)


def compute_highlight_mask(
    gt_image: torch.Tensor,
    latent_hw: tuple[int, int],
    threshold: float = 0.8,
    background_threshold: float = 0.98,
    use_quantile_threshold: bool = False,
    highlight_quantile: float = 0.95,
    min_threshold: float = 0.6,
    max_threshold: float = 0.95,
    quantile_blur_sigma: float = 0.0,
):
    """
    生成与 latent loss 分辨率对齐的二值高光区域 mask。
    """
    _, highlight_mask, _ = compute_highlight_score_map(
        gt_image=gt_image,
        latent_hw=latent_hw,
        threshold=threshold,
        background_threshold=background_threshold,
        use_quantile_threshold=use_quantile_threshold,
        highlight_quantile=highlight_quantile,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        quantile_blur_sigma=quantile_blur_sigma,
    )
    return highlight_mask


def summarize_highlight_loss_metrics(
    per_pixel_loss: torch.Tensor,
    highlight_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
):
    """
    统计高光区域与非高光区域的 loss 表现。

    这里会先把 channel 维做平均，使 loss 语义更接近“每个像素位置的 MSE”。
    """
    spatial_loss = per_pixel_loss.mean(dim=1, keepdim=True).float()
    highlight_mask = highlight_mask.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(highlight_mask)
    else:
        valid_mask = valid_mask.float()
    highlight_mask = highlight_mask * valid_mask
    non_highlight_mask = (valid_mask - highlight_mask).clamp(min=0.0)

    highlight_sum = highlight_mask.sum()
    non_highlight_sum = non_highlight_mask.sum()

    highlight_mse = (spatial_loss * highlight_mask).sum() / highlight_sum.clamp_min(1e-8)
    non_highlight_mse = (spatial_loss * non_highlight_mask).sum() / non_highlight_sum.clamp_min(1e-8)
    highlight_mse_ratio = highlight_mse / non_highlight_mse.clamp_min(1e-8)

    return {
        "highlight_region_ratio": highlight_mask.mean(),
        "highlight_mse": highlight_mse,
        "non_highlight_mse": non_highlight_mse,
        "highlight_mse_ratio": highlight_mse_ratio,
    }


def build_highlight_visualizations(gt_image: torch.Tensor, args, output_hw: tuple[int, int]):
    """
    构造验证阶段要显示的两张图:
    - highlight_mask: 二值高光 mask
    - highlight_weight: 归一化后的高光权重图

    这两个可视化不会参与训练，只是为了让人更直观地确认:
    “模型当前重点关注的区域，是否真的是我们想强调的高光部分。”
    """
    gt_img_01 = (gt_image.float() + 1.0) / 2.0
    luminance = (
        0.299 * gt_img_01[:, 0:1]
        + 0.587 * gt_img_01[:, 1:2]
        + 0.114 * gt_img_01[:, 2:3]
    )
    mask = compute_highlight_mask(
        gt_image=gt_image,
        latent_hw=luminance.shape[-2:],
        threshold=getattr(args, "highlight_threshold", 0.8),
        background_threshold=getattr(args, "foreground_background_threshold", 0.98),
        use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
        highlight_quantile=getattr(args, "highlight_quantile", 0.95),
        min_threshold=getattr(args, "highlight_min_threshold", 0.6),
        max_threshold=getattr(args, "highlight_max_threshold", 0.95),
        quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
    )
    weight_map = compute_highlight_weight_map(
        gt_image=gt_image,
        latent_hw=luminance.shape[-2:],
        threshold=getattr(args, "highlight_threshold", 0.8),
        extra_weight=getattr(args, "highlight_loss_weight", 1.0),
        soft_weighting=getattr(args, "highlight_soft_weighting", False),
        gamma=getattr(args, "highlight_gamma", 2.0),
        background_threshold=getattr(args, "foreground_background_threshold", 0.98),
        use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
        highlight_quantile=getattr(args, "highlight_quantile", 0.95),
        min_threshold=getattr(args, "highlight_min_threshold", 0.6),
        max_threshold=getattr(args, "highlight_max_threshold", 0.95),
        quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
    )
    weight_norm = ((weight_map - 1.0) / max(1e-6, getattr(args, "highlight_loss_weight", 1.0))).clamp(0.0, 1.0)

    mask_rgb = mask.repeat(1, 3, 1, 1)
    weight_rgb = weight_norm.repeat(1, 3, 1, 1)

    if mask_rgb.shape[-2:] != output_hw:
        mask_rgb = F.interpolate(mask_rgb, size=output_hw, mode="nearest")
    if weight_rgb.shape[-2:] != output_hw:
        weight_rgb = F.interpolate(weight_rgb, size=output_hw, mode="bilinear", align_corners=False)

    mask_npy = np.array(mask_rgb.permute([0, 2, 3, 1]).cpu(), dtype=np.float32)
    weight_npy = np.array(weight_rgb.permute([0, 2, 3, 1]).cpu(), dtype=np.float32)
    return mask_npy, weight_npy


def resolve_wandb_project_name(args) -> str:
    """
    优先沿用本地最近一次 wandb run 的项目名，找不到时回退到配置值。
    """
    fallback_project = getattr(args, "tracker_project_name", "train_neural_gaffer_private")
    workspace_root = Path(__file__).resolve().parent
    debug_log_path = workspace_root / "wandb" / "latest-run" / "logs" / "debug.log"

    try:
        if debug_log_path.exists():
            lines = debug_log_path.read_text(errors="ignore").splitlines()
            for line in reversed(lines):
                match = re.search(r"finishing run [^/]+/([^/]+)/[^/\s]+", line)
                if match:
                    return match.group(1)
    except Exception as exc:
        logger.warning("Failed to infer the last wandb project from %s: %s", debug_log_path, exc)

    return fallback_project


def _sanitize_wandb_name_part(value: str) -> str:
    """
    把自由文本压缩成更适合作为 wandb run 名称片段的形式。
    """
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-_.")
    return value


def _format_run_float_token(value: float, scale: int = 100) -> str:
    scaled = int(round(float(value) * scale))
    return str(scaled)


def _next_wandb_daily_index(now_utc: datetime.datetime) -> int:
    wandb_root = Path(__file__).resolve().parent / "wandb"
    date_prefix = now_utc.strftime("run-%Y%m%d_")
    match_count = 0

    if not wandb_root.exists():
        return 1

    for run_dir in wandb_root.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith(date_prefix):
            match_count += 1

    return match_count + 1


def build_wandb_run_name(args) -> str:
    """
    自动生成当前实验的 wandb run 名称。

    命名策略:
    - 只保留必要的实验参数，不再带 output_dir 前缀
    - 日期部分使用 MMDD-当日第几次 的短格式
    """
    if getattr(args, "wandb_run_name", None):
        return args.wandb_run_name

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    parts = []
    use_highlight_weighted_loss = getattr(args, "use_highlight_weighted_loss", False)
    use_image_space_highlight_loss = getattr(args, "use_image_space_highlight_loss", False)
    random_lighting_condition_prob = getattr(args, "random_lighting_condition_prob", 0.1)

    if use_highlight_weighted_loss:
        parts.append("latentsoft" if getattr(args, "highlight_soft_weighting", False) else "latenthard")
    if use_image_space_highlight_loss:
        parts.append("imgsoft" if getattr(args, "highlight_soft_weighting", False) else "imghard")
    if not use_highlight_weighted_loss and not use_image_space_highlight_loss:
        parts.append("plain")

    threshold = getattr(args, "highlight_threshold", None)
    extra_weight = getattr(args, "highlight_loss_weight", None)
    gamma = getattr(args, "highlight_gamma", None)
    quantile_blur_sigma = float(getattr(args, "highlight_quantile_blur_sigma", 0.0) or 0.0)
    image_space_constraint_warmup_steps = int(getattr(args, "image_space_constraint_warmup_steps", 0) or 0)
    highlight_loss_weight_warmup_steps = int(getattr(args, "highlight_loss_weight_warmup_steps", 0) or 0)
    if getattr(args, "highlight_use_quantile_threshold", False):
        parts.append(f"q{_format_run_float_token(getattr(args, 'highlight_quantile', 0.95), scale=100)}")
        if quantile_blur_sigma > 0.0:
            parts.append(f"gb{_format_run_float_token(quantile_blur_sigma, scale=10)}")
    elif threshold is not None:
        parts.append(f"t{_format_run_float_token(threshold, scale=100)}")
    parts.append(f"r{_format_run_float_token(random_lighting_condition_prob, scale=100)}")
    if extra_weight is not None and (use_highlight_weighted_loss or use_image_space_highlight_loss):
        parts.append(f"w{extra_weight:g}")
    if gamma is not None and getattr(args, "highlight_soft_weighting", False):
        parts.append(f"g{gamma:g}")
    if image_space_constraint_warmup_steps > 0 or highlight_loss_weight_warmup_steps > 0:
        if image_space_constraint_warmup_steps == highlight_loss_weight_warmup_steps:
            parts.append(f"wu{int(image_space_constraint_warmup_steps / 1000)}k")
        else:
            if image_space_constraint_warmup_steps > 0:
                parts.append(f"iwu{int(image_space_constraint_warmup_steps / 1000)}k")
            if highlight_loss_weight_warmup_steps > 0:
                parts.append(f"hwu{int(highlight_loss_weight_warmup_steps / 1000)}k")
    if getattr(args, "max_train_steps", None):
        parts.append(f"{int(args.max_train_steps / 1000)}k")

    note = getattr(args, "wandb_run_note", None)
    if note:
        sanitized_note = _sanitize_wandb_name_part(note)
        if sanitized_note:
            parts.append(sanitized_note)

    parts.append(now_utc.strftime("%m%d"))
    parts.append(f"{_next_wandb_daily_index(now_utc):02d}")
    return "-".join(_sanitize_wandb_name_part(part) for part in parts if part)

def _encode_image(image_encoder, image, device, dtype, do_classifier_free_guidance):

    image = image.to(device=device, dtype=dtype)
    image = CLIP_preprocess(image)
    # if not isinstance(image, torch.Tensor):
    #     # 0-255
    #     print("Warning: image is processed by hf's preprocess, which is different from openai original's.")
    #     image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    image_embeddings = image_encoder(image).image_embeds.to(dtype=dtype)
    image_embeddings = image_embeddings.unsqueeze(1)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings.detach()


def _encode_image_without_pose(image_encoder, image, device, dtype, do_classifier_free_guidance):
    img_prompt_embeds = _encode_image(image_encoder, image, device, dtype, False)
    prompt_embeds = img_prompt_embeds
    # follow 0123, add negative prompt, after projection
    if do_classifier_free_guidance:
        negative_prompt = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
    return prompt_embeds

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    loading_kwargs = {
        "low_cpu_mem_usage": True,
        "revision": args.revision
    }
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", **loading_kwargs)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", **loading_kwargs)
    feature_extractor = None #CLIPFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", **loading_kwargs)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", **loading_kwargs)
    
    
    vae.train()
    image_encoder.train()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    # zero init unet conv_in from 8 channels to 16 channels
    conv_in_16 = torch.nn.Conv2d(16, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
    conv_in_16.requires_grad_(False)
    unet.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_16.weight)
    conv_in_16.weight[:,:8,:,:].copy_(unet.conv_in.weight)
    conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    unet.requires_grad_(True)
    unet.train()



    
    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
                vae.enable_tiling()
                logger.info("Enabled xFormers memory efficient attention.")
            except Exception as exc:
                logger.warning(
                    "Failed to enable xFormers memory efficient attention; continuing without it. "
                    "This usually means the installed xFormers kernels do not support the current GPU or dtype. "
                    "Error: %s",
                    exc,
                )
        #else:
            #raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )


    # Enable TF32 for faster training on modern NVIDIA GPUs.
    # Also let cuDNN autotune convolution kernels for our fixed-size 256x256 training pipeline.
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.training_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        [{"params": unet.parameters(), "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )


    # 当前默认策略:
    # - 保留标准 diffusion loss 作为主目标
    # - 默认开启 image-space 高光辅助约束
    # - 默认关闭 latent/noise-space 的高光重加权
    use_highlight_weighted_loss = getattr(args, "use_highlight_weighted_loss", False)
    use_image_space_highlight_loss = getattr(args, "use_image_space_highlight_loss", True)
    if use_highlight_weighted_loss:
        logger.info(
            "Latent/noise-space highlight-weighted diffusion loss enabled with weight=%s warmup_steps=%s threshold=%s soft=%s gamma=%s quantile=%s q=%s min_t=%s max_t=%s blur_sigma=%s",
            getattr(args, "highlight_loss_weight", 1.0),
            getattr(args, "highlight_loss_weight_warmup_steps", 0),
            getattr(args, "highlight_threshold", 0.8),
            getattr(args, "highlight_soft_weighting", False),
            getattr(args, "highlight_gamma", 2.0),
            getattr(args, "highlight_use_quantile_threshold", False),
            getattr(args, "highlight_quantile", 0.95),
            getattr(args, "highlight_min_threshold", 0.6),
            getattr(args, "highlight_max_threshold", 0.95),
            getattr(args, "highlight_quantile_blur_sigma", 0.0),
        )
    else:
        logger.info("Latent/noise-space highlight-weighted diffusion loss disabled")

    if use_image_space_highlight_loss:
        logger.info(
            "Image-space highlight constraint enabled with constraint_weight=%s constraint_warmup_steps=%s highlight_weight=%s highlight_weight_warmup_steps=%s threshold=%s soft=%s gamma=%s quantile=%s q=%s min_t=%s max_t=%s blur_sigma=%s",
            getattr(args, "image_space_constraint_weight", 0.1),
            getattr(args, "image_space_constraint_warmup_steps", 0),
            getattr(args, "highlight_loss_weight", 1.0),
            getattr(args, "highlight_loss_weight_warmup_steps", 0),
            getattr(args, "highlight_threshold", 0.8),
            getattr(args, "highlight_soft_weighting", False),
            getattr(args, "highlight_gamma", 2.0),
            getattr(args, "highlight_use_quantile_threshold", False),
            getattr(args, "highlight_quantile", 0.95),
            getattr(args, "highlight_min_threshold", 0.6),
            getattr(args, "highlight_max_threshold", 0.95),
            getattr(args, "highlight_quantile_blur_sigma", 0.0),
        )
    else:
        logger.info("Image-space highlight constraint disabled")
    logger.info(
        "Training random area-light condition probability=%s",
        getattr(args, "random_lighting_condition_prob", 0.1),
    )
    logger.info(
        "Best pre-window checkpoint tracking enabled until step=%s using held-out mean PSNR",
        getattr(args, "best_checkpoint_until_step", 72000),
    )
    logger.info(
        "Training guards: non_finite_patience=%s collapse_enabled=%s collapse_psnr_threshold=%s collapse_relative_ratio=%s",
        getattr(args, "non_finite_early_stop_patience", 3),
        getattr(args, "early_stop_on_validation_collapse", True),
        getattr(args, "collapse_psnr_threshold", 5.0),
        getattr(args, "collapse_relative_psnr_ratio", 0.25),
    )
    
    # print model info, learnable parameters, non-learnable parameters, total parameters, model size, all in billion
    def print_model_info(model):
        # only rank 0 print
        if accelerator.is_main_process:
            print("="*20)
            # print model class name
            print("model name: ", type(model).__name__)
            # print("model: ", model)
            print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
            print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
            print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
            print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)

    print_model_info(unet)
    print_model_info(vae)
    print_model_info(image_encoder)
    
    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution)),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
 
    
    train_dataset = NeuralGafferTrainingData(
        img_dir = args.train_img_dir,
        lighting_dir = args.train_lighting_dir,
        image_transforms=image_transforms, 
        lighting_per_view=16,
        total_view=12,
        validation=False,
        relighting_only=True,
        image_preprocessed = True,
        dataset_type='training_object_with_seen_envir',
        random_lighting_condition_prob=args.random_lighting_condition_prob,
        )
    
    # validate seen training object with unseen lighting, and the input images of are rendered with unseen lighting under unseen camera poses
    training_dataset_unseen_lighting = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/unseen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/unseen_lighting/',
        lighting_dir = os.path.join(args.val_lighting_dir, 'unseen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'unseen_lighting'),
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        relighting_only=True,
        image_preprocessed = False,
        dataset_type='training_object_with_unseen_envir'
        )   
    
    # validate unseen object with unseen lighting, and the input images of the unseen object are rendered with random area lighting 
    validation_dataset_unseen_lighting_random_light_condition = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/unseen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/unseen_lighting/', 
        lighting_dir = os.path.join(args.val_lighting_dir, 'unseen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'unseen_lighting'),
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        image_preprocessed = True,  
        relighting_only=True,
        dataset_type='unseen_object_with_random_area_light_condition'
        )       


    validation_dataset_seen_lighting = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/seen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/seen_lighting/', 
        lighting_dir = os.path.join(args.val_lighting_dir, 'seen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'seen_lighting'),   
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        image_preprocessed = True,  
        relighting_only=True,
        dataset_type='unseen_object_with_seen_envir'
        ) 
    
    validation_dataset_unseen_lighting = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/unseen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/unseen_lighting/', 
        lighting_dir = os.path.join(args.val_lighting_dir, 'unseen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'unseen_lighting'),
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        image_preprocessed = True,  
        relighting_only=True,
        dataset_type='unseen_object_with_unseen_envir'
        )   
    
       
    train_prefetch_factor = 2 if args.dataloader_num_workers and args.dataloader_num_workers > 0 else None
    train_persistent_workers = bool(args.dataloader_num_workers and args.dataloader_num_workers > 0)
    eval_persistent_workers = True

    # for training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.training_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=train_persistent_workers,
        prefetch_factor=train_prefetch_factor,
    )

    validation_dataloader_random_light_condition = torch.utils.data.DataLoader(
        validation_dataset_unseen_lighting_random_light_condition,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=eval_persistent_workers,
        prefetch_factor=2,
    )

    # for validation set logs
    training_dataloader_unseen_lighting = torch.utils.data.DataLoader(
        training_dataset_unseen_lighting,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=eval_persistent_workers,
        prefetch_factor=2,
    )

    validation_dataloader_seen_lighting = torch.utils.data.DataLoader(
        validation_dataset_seen_lighting,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=eval_persistent_workers,
        prefetch_factor=2,
    )

    # for unseen objects validation set logs    
    validation_dataloader_unseen_lighting = torch.utils.data.DataLoader(
        validation_dataset_unseen_lighting,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=eval_persistent_workers,
        prefetch_factor=2,
    )

    # for training set logs
    train_log_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=eval_persistent_workers,
        prefetch_factor=2,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, train_log_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, train_log_dataloader, lr_scheduler
    )
    

    
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, image_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_project_name = resolve_wandb_project_name(args)
        tracker_run_name = build_wandb_run_name(args)
        wandb_resume_id = getattr(args, "wandb_resume_id", None) or os.environ.get("WANDB_RUN_ID")
        wandb_resume_mode = getattr(args, "wandb_resume_mode", None) or os.environ.get("WANDB_RESUME")
        tracker_config["tracker_project_name"] = tracker_project_name
        tracker_config["wandb_run_name"] = tracker_run_name
        tracker_config["wandb_resume_id"] = wandb_resume_id
        tracker_config["wandb_resume_mode"] = wandb_resume_mode
        logger.info("Using wandb project: %s", tracker_project_name)
        logger.info("Using wandb run name: %s", tracker_run_name)
        if wandb_resume_id:
            logger.info("Resuming wandb run id: %s (mode=%s)", wandb_resume_id, wandb_resume_mode or "allow")
        wandb_init_kwargs = {}
        if not wandb_resume_id or getattr(args, "wandb_run_name", None):
            wandb_init_kwargs["name"] = tracker_run_name
        if wandb_resume_id:
            wandb_init_kwargs["id"] = wandb_resume_id
            wandb_init_kwargs["resume"] = wandb_resume_mode or "allow"
        accelerator.init_trackers(
            tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    # Train!
    total_batch_size = args.training_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.training_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f" conditioning_dropout_prob = {args.conditioning_dropout_prob}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    best_pre72k_metadata = load_best_pre72k_metadata(args.output_dir) if args.output_dir else None
    best_pre72k_score = float("-inf")
    best_pre72k_step = None
    best_pre72k_path = None
    if best_pre72k_metadata is not None:
        metadata_score = best_pre72k_metadata.get("score")
        metadata_step = best_pre72k_metadata.get("step")
        metadata_path = best_pre72k_metadata.get("checkpoint_path")
        if isinstance(metadata_score, (int, float)) and math.isfinite(float(metadata_score)):
            best_pre72k_score = float(metadata_score)
        if isinstance(metadata_step, int):
            best_pre72k_step = metadata_step
        if isinstance(metadata_path, str) and metadata_path:
            best_pre72k_path = metadata_path
        logger.info(
            "Loaded existing best pre-window checkpoint metadata: step=%s score=%s path=%s",
            best_pre72k_step,
            best_pre72k_score if math.isfinite(best_pre72k_score) else None,
            best_pre72k_path,
        )

    non_finite_step_streak = 0
    early_stop_triggered = False
    early_stop_reason = None

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        for step, batch in enumerate(train_dataloader):
            skipped_update = False
            skip_reason = None
            train_loss_scalar = None
            image_space_aux_skipped = False
            
            with accelerator.accumulate(unet):
                # Convert images to latent space
                input_image = batch["image_cond"].to(dtype=weight_dtype)
                relighting_image_group1 = batch["image_target"].to(dtype=weight_dtype)
                relighting_image_group2 = batch["image_another_target"].to(dtype=weight_dtype)
                pose = batch["T"].to(dtype=weight_dtype)
                pose = torch.cat([pose, pose], dim=0)
                input_image = torch.cat((input_image, input_image), dim=0)
                gt_image = torch.cat((relighting_image_group1, relighting_image_group2), dim=0)
                
                # environment map target
                target_envir_map_ldr_group1 = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
                target_envir_map_hdr_group1 = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
                target_envir_map_ldr_group2 = batch["envir_map_another_target_ldr"].to(dtype=weight_dtype)
                target_envir_map_hdr_group2 = batch["envir_map_another_target_hdr"].to(dtype=weight_dtype)
                target_envir_map_ldr = torch.cat((target_envir_map_ldr_group1, target_envir_map_ldr_group2), dim=0)
                target_envir_map_hdr = torch.cat((target_envir_map_hdr_group1, target_envir_map_hdr_group2), dim=0)
                

                # pose = torch.cat([pose, target_orientation], dim=-1)

                gt_latents = vae.encode(gt_image).latent_dist.sample().detach()
                gt_latents = gt_latents * vae.config.scaling_factor # follow zero123, only target image latent is scaled

                img_latents = vae.encode(input_image).latent_dist.mode().detach()   
                target_envir_map_ldr_latents = vae.encode(target_envir_map_ldr).latent_dist.sample().detach()
                target_envir_map_hdr_latents = vae.encode(target_envir_map_hdr).latent_dist.mode().detach()
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(gt_latents)
                bsz = gt_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(gt_latents.to(dtype=torch.float32), noise.to(dtype=torch.float32), timesteps).to(dtype=img_latents.dtype)
                if do_classifier_free_guidance:  # support classifier-free guidance, randomly drop out 5%
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    random_p = torch.rand(bsz, device=gt_latents.device)
                    
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)

                    img_prompt_embeds = _encode_image(image_encoder, input_image, gt_latents.device, gt_latents.dtype, False)

                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(img_prompt_embeds).detach()
                    img_prompt_embeds = torch.where(prompt_mask, null_conditioning, img_prompt_embeds)

                    prompt_embeds = img_prompt_embeds

                    # Sample masks for the input images.
                    image_mask_dtype = img_latents.dtype
                    image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    img_latents = image_mask * img_latents
                    target_envir_map_ldr_latents = image_mask * target_envir_map_ldr_latents
                    target_envir_map_hdr_latents = image_mask * target_envir_map_hdr_latents
                else:
                    # Get the image_with_pose embedding for conditioning
                    prompt_embeds = _encode_image_without_pose(image_encoder, input_image, gt_latents.device, weight_dtype, False)


                # latent_model_input = torch.cat([noisy_latents, img_latents], dim=1)
                latent_model_input = torch.cat([noisy_latents, img_latents, target_envir_map_hdr_latents, target_envir_map_ldr_latents], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                # per_pixel_loss:
                # - 标准 diffusion 训练里最核心的逐像素噪声 MSE
                # - 形状与 model_pred / target 相同
                diffusion_per_pixel_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                latent_highlight_mask = compute_highlight_mask(
                    gt_image=gt_image,
                    latent_hw=diffusion_per_pixel_loss.shape[-2:],
                    threshold=getattr(args, "highlight_threshold", 0.8),
                    background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                    use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
                    highlight_quantile=getattr(args, "highlight_quantile", 0.95),
                    min_threshold=getattr(args, "highlight_min_threshold", 0.6),
                    max_threshold=getattr(args, "highlight_max_threshold", 0.95),
                    quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
                ).to(device=diffusion_per_pixel_loss.device, dtype=diffusion_per_pixel_loss.dtype)
                latent_foreground_mask = compute_foreground_mask(
                    gt_image=gt_image,
                    latent_hw=diffusion_per_pixel_loss.shape[-2:],
                    background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                ).to(device=diffusion_per_pixel_loss.device, dtype=diffusion_per_pixel_loss.dtype)
                latent_highlight_metric_tensors = summarize_highlight_loss_metrics(
                    diffusion_per_pixel_loss,
                    latent_highlight_mask,
                    valid_mask=latent_foreground_mask,
                )
                logging_steps = max(1, getattr(args, "logging_steps", 50))
                should_log_highlight_metrics = global_step % logging_steps == 0

                effective_highlight_loss_weight = get_warmup_scaled_weight(
                    getattr(args, "highlight_loss_weight", 1.0),
                    global_step=global_step,
                    warmup_steps=getattr(args, "highlight_loss_weight_warmup_steps", 0),
                )
                effective_image_space_constraint_weight = get_warmup_scaled_weight(
                    getattr(args, "image_space_constraint_weight", 0.1),
                    global_step=global_step,
                    warmup_steps=getattr(args, "image_space_constraint_warmup_steps", 0),
                )

                weight_map = None
                if use_highlight_weighted_loss:
                    # 根据真实目标图生成一个与 latent 分辨率一致的空间权重图。
                    # 这样模型在学习噪声时，会对高光区域承担更大的误差惩罚。
                    weight_map = compute_highlight_weight_map(
                        gt_image=gt_image,
                        latent_hw=diffusion_per_pixel_loss.shape[-2:],
                        threshold=getattr(args, "highlight_threshold", 0.8),
                        extra_weight=effective_highlight_loss_weight,
                        soft_weighting=getattr(args, "highlight_soft_weighting", False),
                        gamma=getattr(args, "highlight_gamma", 2.0),
                        background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                        use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
                        highlight_quantile=getattr(args, "highlight_quantile", 0.95),
                        min_threshold=getattr(args, "highlight_min_threshold", 0.6),
                        max_threshold=getattr(args, "highlight_max_threshold", 0.95),
                        quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
                    ).to(device=diffusion_per_pixel_loss.device, dtype=diffusion_per_pixel_loss.dtype)
                    # weighted_loss: 每个位置的 loss 乘以空间权重
                    weighted_loss = diffusion_per_pixel_loss * weight_map
                    # norm: 用于归一化，避免“权重越大，总 loss 就机械变大太多”
                    # diffusion_per_pixel_loss.shape[1] 对应通道数 C
                    norm = weight_map.sum() * diffusion_per_pixel_loss.shape[1]
                    diffusion_loss = weighted_loss.sum() / norm.clamp_min(1e-6)
                else:
                    diffusion_loss = diffusion_per_pixel_loss.mean()

                image_space_loss = torch.zeros((), device=diffusion_loss.device, dtype=diffusion_loss.dtype)
                image_weight_map = None
                active_highlight_metric_tensors = latent_highlight_metric_tensors
                highlight_logs_prefix = ""

                if use_image_space_highlight_loss:
                    pred_x0_latents = predict_x0_from_model_pred(
                        noise_scheduler,
                        model_pred.float(),
                        noisy_latents.float(),
                        timesteps,
                    )
                    if tensor_is_finite(pred_x0_latents):
                        pred_x0_image = decode_latents_to_image(
                            vae,
                            pred_x0_latents,
                            output_dtype=gt_image.dtype,
                        )
                        if tensor_is_finite(pred_x0_image):
                            image_per_pixel_loss = F.mse_loss(pred_x0_image.float(), gt_image.float(), reduction="none")
                            image_weight_map = compute_highlight_weight_map(
                                gt_image=gt_image,
                                latent_hw=image_per_pixel_loss.shape[-2:],
                                threshold=getattr(args, "highlight_threshold", 0.8),
                                extra_weight=effective_highlight_loss_weight,
                                soft_weighting=getattr(args, "highlight_soft_weighting", False),
                                gamma=getattr(args, "highlight_gamma", 2.0),
                                background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                                use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
                                highlight_quantile=getattr(args, "highlight_quantile", 0.95),
                                min_threshold=getattr(args, "highlight_min_threshold", 0.6),
                                max_threshold=getattr(args, "highlight_max_threshold", 0.95),
                                quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
                            ).to(device=image_per_pixel_loss.device, dtype=image_per_pixel_loss.dtype)
                            image_weighted_loss = image_per_pixel_loss * image_weight_map
                            image_norm = image_weight_map.sum() * image_per_pixel_loss.shape[1]
                            image_space_loss = image_weighted_loss.sum() / image_norm.clamp_min(1e-6)

                            image_highlight_mask = compute_highlight_mask(
                                gt_image=gt_image,
                                latent_hw=image_per_pixel_loss.shape[-2:],
                                threshold=getattr(args, "highlight_threshold", 0.8),
                                background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                                use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", False),
                                highlight_quantile=getattr(args, "highlight_quantile", 0.95),
                                min_threshold=getattr(args, "highlight_min_threshold", 0.6),
                                max_threshold=getattr(args, "highlight_max_threshold", 0.95),
                                quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 0.0),
                            ).to(device=image_per_pixel_loss.device, dtype=image_per_pixel_loss.dtype)
                            image_foreground_mask = compute_foreground_mask(
                                gt_image=gt_image,
                                latent_hw=image_per_pixel_loss.shape[-2:],
                                background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                            ).to(device=image_per_pixel_loss.device, dtype=image_per_pixel_loss.dtype)
                            active_highlight_metric_tensors = summarize_highlight_loss_metrics(
                                image_per_pixel_loss,
                                image_highlight_mask,
                                valid_mask=image_foreground_mask,
                            )
                            highlight_logs_prefix = "image_space/"
                        else:
                            image_space_aux_skipped = True
                            logger.warning(
                                "Skipping image-space auxiliary loss at global_step=%s epoch=%s batch_step=%s because decoded pred_x0 image is non-finite.",
                                global_step,
                                epoch,
                                step,
                            )
                    else:
                        image_space_aux_skipped = True
                        logger.warning(
                            "Skipping image-space auxiliary loss at global_step=%s epoch=%s batch_step=%s because pred_x0 latents are non-finite.",
                            global_step,
                            epoch,
                            step,
                        )

                loss = diffusion_loss + effective_image_space_constraint_weight * image_space_loss

                if not tensor_is_finite(model_pred):
                    skipped_update = True
                    skip_reason = "non_finite_model_pred"
                elif not tensor_is_finite(diffusion_loss):
                    skipped_update = True
                    skip_reason = "non_finite_diffusion_loss"
                elif not tensor_is_finite(image_space_loss):
                    skipped_update = True
                    skip_reason = "non_finite_image_space_loss"
                elif not tensor_is_finite(loss):
                    skipped_update = True
                    skip_reason = "non_finite_total_loss"

                if should_log_highlight_metrics:
                    reduced_highlight_metrics = {
                        key: accelerator.reduce(value.detach(), reduction="mean").item()
                        for key, value in active_highlight_metric_tensors.items()
                    }
                    highlight_logs = dict(reduced_highlight_metrics)
                    if highlight_logs_prefix:
                        highlight_logs.update(
                            {
                                f"{highlight_logs_prefix}{key}": value
                                for key, value in reduced_highlight_metrics.items()
                            }
                        )
                    highlight_logs.update(
                        {
                            "loss_diffusion": accelerator.reduce(diffusion_loss.detach(), reduction="mean").item(),
                            "loss_image_space_constraint": accelerator.reduce(image_space_loss.detach(), reduction="mean").item(),
                            "loss_image_space_constraint_weighted": accelerator.reduce(
                                (effective_image_space_constraint_weight * image_space_loss).detach(),
                                reduction="mean",
                            ).item(),
                            "image_space/effective_constraint_weight": effective_image_space_constraint_weight,
                            "highlight/effective_loss_weight": effective_highlight_loss_weight,
                            "image_space/constraint_warmup_scale": linear_warmup_scale(
                                global_step,
                                getattr(args, "image_space_constraint_warmup_steps", 0),
                            ),
                            "highlight/loss_weight_warmup_scale": linear_warmup_scale(
                                global_step,
                                getattr(args, "highlight_loss_weight_warmup_steps", 0),
                            ),
                            "train_guard/image_space_aux_skipped": int(image_space_aux_skipped),
                        }
                    )
                    if use_highlight_weighted_loss and weight_map is not None:
                        highlight_logs = {
                            **highlight_logs,
                            "highlight_weight/mean": accelerator.reduce(weight_map.mean().detach(), reduction="mean").item(),
                            "highlight_weight/max": accelerator.reduce(weight_map.max().detach(), reduction="mean").item(),
                            "highlight_weight/highlight_fraction": accelerator.reduce(
                                (weight_map > 1.0).float().mean().detach(),
                                reduction="mean",
                            ).item(),
                        }
                    if image_weight_map is not None:
                        highlight_logs = {
                            **highlight_logs,
                            "image_space/highlight_weight_mean": accelerator.reduce(
                                image_weight_map.mean().detach(),
                                reduction="mean",
                            ).item(),
                        }
                    sanitized_highlight_logs, skipped_highlight_logs = sanitize_log_dict(highlight_logs)
                    if skipped_highlight_logs:
                        logger.warning(
                            "Dropped %s non-finite highlight log entries at global_step=%s: %s",
                            len(skipped_highlight_logs),
                            global_step,
                            ", ".join(sorted(skipped_highlight_logs.keys())),
                        )
                    if accelerator.is_main_process and sanitized_highlight_logs:
                        accelerator.log(sanitized_highlight_logs, step=global_step)

                if skipped_update:
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                else:
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.unscale_gradients(optimizer=optimizer)

                    if model_has_non_finite_gradients(unet.parameters()):
                        skipped_update = True
                        skip_reason = "non_finite_gradients"
                        optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                        reset_amp_scaler_after_skipped_step(accelerator)
                    else:
                        if accelerator.sync_gradients and args.max_grad_norm is not None and args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                        train_loss_scalar = float(loss.detach().item())
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if skipped_update:
                non_finite_step_streak += 1
                logger.warning(
                    "Skipped optimizer update at global_step=%s epoch=%s batch_step=%s because %s (streak=%s).",
                    global_step,
                    epoch,
                    step,
                    skip_reason,
                    non_finite_step_streak,
                )
                guard_logs = {
                    "train_guard/skipped_update": 1,
                    "train_guard/non_finite_step_streak": non_finite_step_streak,
                    "train_guard/image_space_aux_skipped_step": int(image_space_aux_skipped),
                }
                sanitized_guard_logs, skipped_guard_logs = sanitize_log_dict(guard_logs)
                if skipped_guard_logs:
                    logger.warning(
                        "Dropped %s non-finite guard log entries at global_step=%s: %s",
                        len(skipped_guard_logs),
                        global_step,
                        ", ".join(sorted(skipped_guard_logs.keys())),
                    )
                if sanitized_guard_logs:
                    accelerator.log(sanitized_guard_logs, step=global_step)

                if non_finite_step_streak >= getattr(args, "non_finite_early_stop_patience", 3):
                    early_stop_triggered = True
                    early_stop_reason = (
                        f"Reached non-finite early-stop patience: {non_finite_step_streak} skipped updates "
                        f"at global_step={global_step}."
                    )
                    logger.warning(early_stop_reason)
            else:
                non_finite_step_streak = 0

            if accelerator.sync_gradients and not skipped_update:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    step_log = {}
                    checkpoint_saved_this_step = None
                    
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        checkpoint_saved_this_step = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(checkpoint_saved_this_step)
                        logger.info(f"Saved state to {checkpoint_saved_this_step}")


                    initial_val_step = getattr(args, 'initial_validation_step', -1)
                    initial_val_trigger = (initial_val_step > 0 and global_step == (initial_val_step + initial_global_step))
                    
                    if validation_dataloader_random_light_condition is not None and (global_step % args.validation_steps == 0 or initial_val_trigger):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            validation_dataloader_random_light_condition,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='unseen_object_with_random_area_light_condition',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                        

                    if validation_dataloader_unseen_lighting is not None and (global_step % args.validation_steps == 0 or initial_val_trigger):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            validation_dataloader_unseen_lighting,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='unseen_object_with_unseen_envir',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                        
                    if validation_dataloader_seen_lighting is not None and (global_step % args.validation_steps == 0 or initial_val_trigger):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            validation_dataloader_seen_lighting,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='unseen_object_with_seen_envir',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                    if training_dataloader_unseen_lighting is not None and (global_step % args.validation_steps == 0 or initial_val_trigger):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            training_dataloader_unseen_lighting,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='training_object_with_unseen_envir',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)   
                          
                            
                    if train_log_dataloader is not None and (global_step % args.validation_steps == 0 or initial_val_trigger):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        train_image_logs, temp_log = log_validation(
                            train_log_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            'train',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)

                    validation_score, validation_metric_keys = extract_validation_psnr_score(step_log)
                    validation_pred_mean = extract_validation_pred_mean(step_log)
                    if validation_score is not None:
                        step_log["train_guard/heldout_mean_psnr"] = validation_score
                    if validation_pred_mean is not None:
                        step_log["train_guard/heldout_min_pred_mean"] = validation_pred_mean

                    best_checkpoint_until_step = getattr(args, "best_checkpoint_until_step", 72000)
                    if (
                        validation_score is not None
                        and global_step <= best_checkpoint_until_step
                        and validation_score > best_pre72k_score
                    ):
                        best_pre72k_score = validation_score
                        best_pre72k_step = global_step
                        best_pre72k_path = save_best_pre72k_checkpoint(
                            accelerator=accelerator,
                            args=args,
                            global_step=global_step,
                            score=validation_score,
                            metric_keys=validation_metric_keys,
                        )

                    if best_pre72k_step is not None and math.isfinite(best_pre72k_score):
                        step_log["train_guard/best_pre72k_score"] = best_pre72k_score
                        step_log["train_guard/best_pre72k_step"] = best_pre72k_step

                    if (
                        getattr(args, "early_stop_on_validation_collapse", True)
                        and global_step > best_checkpoint_until_step
                        and validation_score is not None
                    ):
                        collapse_threshold = getattr(args, "collapse_psnr_threshold", 5.0)
                        if math.isfinite(best_pre72k_score):
                            collapse_threshold = max(
                                collapse_threshold,
                                best_pre72k_score * getattr(args, "collapse_relative_psnr_ratio", 0.25),
                            )
                        collapse_detected = (not math.isfinite(validation_score)) or (validation_score < collapse_threshold)
                        if validation_pred_mean is not None and validation_pred_mean <= 1e-4:
                            collapse_detected = True

                        if collapse_detected:
                            early_stop_triggered = True
                            early_stop_reason = (
                                f"Validation collapse detected at step={global_step}: "
                                f"heldout_mean_psnr={validation_score:.4f}, "
                                f"heldout_min_pred_mean={validation_pred_mean}, "
                                f"collapse_threshold={collapse_threshold:.4f}, "
                                f"best_pre72k_step={best_pre72k_step}, best_pre72k_score={best_pre72k_score:.4f}."
                            )
                            logger.warning(early_stop_reason)
                            step_log["train_guard/early_stop_triggered"] = 1

                            if checkpoint_saved_this_step and os.path.isdir(checkpoint_saved_this_step):
                                shutil.rmtree(checkpoint_saved_this_step, ignore_errors=True)
                                logger.info(
                                    "Removed checkpoint %s because validation had already collapsed at this step.",
                                    checkpoint_saved_this_step,
                                )

                    sanitized_step_log, skipped_step_log = sanitize_log_dict(step_log)
                    if skipped_step_log:
                        logger.warning(
                            "Dropped %s non-finite validation log entries at global_step=%s: %s",
                            len(skipped_step_log),
                            global_step,
                            ", ".join(sorted(skipped_step_log.keys())),
                        )
                    if sanitized_step_log:
                        accelerator.log(sanitized_step_log, step=global_step)

            early_stop_flag = torch.tensor(
                1.0 if early_stop_triggered else 0.0,
                device=accelerator.device,
            )
            early_stop_triggered = bool(accelerator.reduce(early_stop_flag, reduction="sum").item() > 0)
            if early_stop_triggered and early_stop_reason is None:
                early_stop_reason = "Early stop triggered on another process."

            if train_loss_scalar is not None:
                loss_epoch += train_loss_scalar
                num_train_elems += 1

            loss_epoch_value = loss_epoch / num_train_elems if num_train_elems > 0 else None
            log_payload = {
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                "train_guard/non_finite_step_streak": non_finite_step_streak,
                "train_guard/image_space_aux_skipped_step": int(image_space_aux_skipped),
            }
            if train_loss_scalar is not None:
                log_payload["loss"] = train_loss_scalar
            if loss_epoch_value is not None:
                log_payload["loss_epoch"] = loss_epoch_value
            if skipped_update:
                log_payload["train_guard/skipped_update"] = 1

            sanitized_logs, skipped_logs = sanitize_log_dict(log_payload)
            if skipped_logs:
                logger.warning(
                    "Dropped %s non-finite step log entries at global_step=%s: %s",
                    len(skipped_logs),
                    global_step,
                    ", ".join(sorted(skipped_logs.keys())),
                )
            if sanitized_logs:
                accelerator.log(sanitized_logs, step=global_step)

            progress_logs = {
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
            if train_loss_scalar is not None:
                progress_logs["loss"] = train_loss_scalar
            if loss_epoch_value is not None:
                progress_logs["loss_epoch"] = loss_epoch_value
            if skipped_update:
                progress_logs["guard"] = "skip"

            progress_bar.set_postfix(**progress_logs)

            if early_stop_triggered or global_step >= args.max_train_steps:
                break

        if early_stop_triggered or global_step >= args.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        guard_status_path = os.path.join(args.output_dir, "training_guard_status.json")
        with open(guard_status_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "early_stop_triggered": early_stop_triggered,
                    "early_stop_reason": early_stop_reason,
                    "best_pre72k_step": best_pre72k_step,
                    "best_pre72k_score": best_pre72k_score if math.isfinite(best_pre72k_score) else None,
                    "best_pre72k_path": best_pre72k_path,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if early_stop_triggered and best_pre72k_path and os.path.isdir(best_pre72k_path):
            logger.warning(
                "Early stop was triggered. Skipping final pipeline export from the current in-memory model and keeping the preserved checkpoint instead: %s",
                best_pre72k_path,
            )
            accelerator.end_training()
            return

        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        # unet.save_pretrained(args.output_dir)

        

        pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            image_encoder=accelerator.unwrap_model(image_encoder),
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()




if __name__ == "__main__":
    args = parse_args()
    main(args)
