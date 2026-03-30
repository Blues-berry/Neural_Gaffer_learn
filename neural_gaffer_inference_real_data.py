import os
import json
import contextlib
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from dataset.dataset_relighting_eval_real import Relighting_Data

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

import torchvision
import torch.nn.functional as F

logger = get_logger(__name__)

from parse_args import parse_args


def tensor_bchw_to_numpy_bhwc_01(tensor: torch.Tensor) -> np.ndarray:
    return 0.5 * (np.array(tensor.detach().permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)


def ensure_pil_image(image_like):
    if isinstance(image_like, Image.Image):
        return image_like
    if isinstance(image_like, np.ndarray):
        if image_like.dtype != np.uint8:
            image_like = np.clip(image_like * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(image_like)
    raise TypeError(f"Unsupported image type for PIL conversion: {type(image_like)}")


def single_channel_to_rgb_pil(tensor_b1hw: torch.Tensor, output_hw: tuple[int, int] | None = None) -> Image.Image:
    tensor = tensor_b1hw.detach().float().cpu()
    if output_hw is not None and tensor.shape[-2:] != output_hw:
        tensor = F.interpolate(tensor, size=output_hw, mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0).squeeze(0)
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max().clamp_min(1e-6)
    npy = (tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.stack([npy, npy, npy], axis=-1)
    return Image.fromarray(rgb)


def make_abs_diff_heatmap_pil(
    image_a_bchw: torch.Tensor,
    image_b_bchw: torch.Tensor,
    foreground_mask_b1hw: torch.Tensor | None = None,
) -> Image.Image:
    diff = (image_b_bchw.float() - image_a_bchw.float()).abs().mean(dim=1, keepdim=True)
    if foreground_mask_b1hw is not None:
        diff = diff * foreground_mask_b1hw.float()
    diff = diff / diff.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    diff = diff.squeeze(0).squeeze(0).cpu().numpy()
    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(diff * 255.0, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip((diff ** 0.7) * 180.0, 0, 255).astype(np.uint8)
    return Image.fromarray(heat)


def make_signed_brightness_delta_pil(
    image_a_bchw: torch.Tensor,
    image_b_bchw: torch.Tensor,
    foreground_mask_b1hw: torch.Tensor | None = None,
) -> Image.Image:
    a_01 = (image_a_bchw.float() + 1.0) / 2.0
    b_01 = (image_b_bchw.float() + 1.0) / 2.0
    lum_a = 0.299 * a_01[:, 0:1] + 0.587 * a_01[:, 1:2] + 0.114 * a_01[:, 2:3]
    lum_b = 0.299 * b_01[:, 0:1] + 0.587 * b_01[:, 1:2] + 0.114 * b_01[:, 2:3]
    delta = lum_b - lum_a
    if foreground_mask_b1hw is not None:
        delta = delta * foreground_mask_b1hw.float()
    scale = delta.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    delta = delta / scale
    pos = torch.clamp(delta, min=0.0)
    neg = torch.clamp(-delta, min=0.0)
    rgb = torch.cat([pos, neg, neg * 0.35], dim=1)
    rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))


def make_side_by_side_pil(left_pil: Image.Image, right_pil: Image.Image) -> Image.Image:
    left = left_pil.convert("RGB")
    right = right_pil.convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def compute_foreground_mask_for_inference(image_bchw: torch.Tensor, background_threshold: float = 0.98) -> torch.Tensor:
    image_01 = (image_bchw.float() + 1.0) / 2.0
    return (image_01.amin(dim=1, keepdim=True) < background_threshold).float()


def compute_highlight_local_mean_for_inference(
    luminance: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    local_kernel_size: int = 15,
    eps: float = 1e-6,
) -> torch.Tensor:
    kernel_size = max(int(local_kernel_size or 0), 1)
    if kernel_size <= 1:
        return luminance
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2

    if foreground_mask is None:
        padded_luminance = F.pad(luminance, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(padded_luminance, kernel_size=kernel_size, stride=1, padding=0)

    foreground_mask = foreground_mask.to(device=luminance.device, dtype=luminance.dtype)
    masked_luminance = luminance * foreground_mask
    kernel_area = float(kernel_size * kernel_size)
    local_sum = F.avg_pool2d(masked_luminance, kernel_size=kernel_size, stride=1, padding=padding) * kernel_area
    local_weight = F.avg_pool2d(foreground_mask, kernel_size=kernel_size, stride=1, padding=padding) * kernel_area
    safe_local_mean = local_sum / local_weight.clamp_min(float(eps))
    return torch.where(local_weight > float(eps), safe_local_mean, luminance)


def compute_highlight_diagnostics_for_inference(
    image_bchw: torch.Tensor,
    background_threshold: float = 0.98,
    relative_mode: str = "difference",
    local_kernel_size: int = 15,
    relative_eps: float = 1e-4,
    use_quantile_threshold: bool = True,
    highlight_quantile: float = 0.88,
    min_threshold: float = 0.02,
    max_threshold: float = 0.2,
    quantile_blur_sigma: float = 1.0,
):
    image_01 = (image_bchw.float() + 1.0) / 2.0
    foreground_mask = compute_foreground_mask_for_inference(
        image_bchw,
        background_threshold=background_threshold,
    )
    luminance = (
        0.299 * image_01[:, 0:1]
        + 0.587 * image_01[:, 1:2]
        + 0.114 * image_01[:, 2:3]
    )
    masked_foreground = image_01 * foreground_mask + (1.0 - foreground_mask)

    blur_kernel_size = max(3, int(2 * round(float(quantile_blur_sigma or 0.0) * 3) + 1))
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    if float(quantile_blur_sigma or 0.0) > 0.0:
        blurred_luminance = torchvision.transforms.functional.gaussian_blur(
            luminance,
            kernel_size=[blur_kernel_size, blur_kernel_size],
            sigma=float(quantile_blur_sigma),
        )
    else:
        blurred_luminance = luminance

    mode = str(relative_mode or "none").lower()
    if mode == "none":
        measure_map = luminance
        quantile_measure_map = blurred_luminance
        reference_map = torch.zeros_like(luminance)
    else:
        local_mean = compute_highlight_local_mean_for_inference(
            luminance,
            foreground_mask=foreground_mask,
            local_kernel_size=local_kernel_size,
            eps=relative_eps,
        )
        quantile_local_mean = compute_highlight_local_mean_for_inference(
            blurred_luminance,
            foreground_mask=foreground_mask,
            local_kernel_size=local_kernel_size,
            eps=relative_eps,
        )
        if mode == "difference":
            measure_map = luminance - local_mean
            quantile_measure_map = blurred_luminance - quantile_local_mean
            reference_map = local_mean
        elif mode == "ratio":
            measure_map = luminance / local_mean.clamp_min(float(relative_eps))
            quantile_measure_map = blurred_luminance / quantile_local_mean.clamp_min(float(relative_eps))
            reference_map = local_mean
        else:
            raise ValueError(f"Unsupported highlight_relative_mode: {relative_mode}")

    threshold_values = []
    for idx in range(image_bchw.shape[0]):
        fg_values = quantile_measure_map[idx][foreground_mask[idx] > 0.5]
        if fg_values.numel() == 0:
            threshold_value = float(min_threshold)
        elif use_quantile_threshold:
            threshold_value = torch.quantile(fg_values, float(highlight_quantile)).item()
        else:
            threshold_value = float(min_threshold)
        threshold_values.append(float(np.clip(threshold_value, float(min_threshold), float(max_threshold))))
    threshold_scalar = torch.tensor(
        threshold_values,
        device=image_bchw.device,
        dtype=image_bchw.dtype,
    ).view(-1, 1, 1, 1)

    if mode == "none":
        quantile_threshold_map = threshold_scalar.expand_as(luminance)
    elif mode == "difference":
        quantile_threshold_map = reference_map + threshold_scalar
    else:
        quantile_threshold_map = reference_map * threshold_scalar

    quantile_excess = torch.relu(measure_map - threshold_scalar) * foreground_mask
    quantile_score = quantile_excess / quantile_excess.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    highlight_mask = ((measure_map >= threshold_scalar) * foreground_mask).float()
    highlight_extract = image_01 * highlight_mask + (1.0 - highlight_mask)

    local_relative_vis = measure_map * foreground_mask
    local_relative_flat = local_relative_vis.view(local_relative_vis.shape[0], -1)
    local_relative_min = local_relative_flat.min(dim=1).values.view(-1, 1, 1, 1)
    local_relative_max = local_relative_flat.max(dim=1).values.view(-1, 1, 1, 1)
    local_relative_vis = (local_relative_vis - local_relative_min) / (local_relative_max - local_relative_min).clamp_min(1e-6)

    return {
        "explicit_mask_rgb": foreground_mask.repeat(1, 3, 1, 1),
        "masked_foreground_rgb": masked_foreground,
        "local_relative_rgb": local_relative_vis.repeat(1, 3, 1, 1),
        "gaussian_blur_rgb": blurred_luminance.repeat(1, 3, 1, 1),
        "quantile_estimation_rgb": quantile_threshold_map.clamp(0.0, 1.0).repeat(1, 3, 1, 1),
        "quantile_score_rgb": quantile_score.repeat(1, 3, 1, 1),
        "quantile_highlight_mask_rgb": highlight_mask.repeat(1, 3, 1, 1),
        "foreground_quantile_highlight_extract_rgb": highlight_extract,
    }


def save_stage_outputs(
    save_dir: str,
    input_image_name: str,
    target_lighting_name: str,
    target_view_idx: int,
    input_image_pil: Image.Image,
    target_envmap_hdr_pil: Image.Image,
    target_envmap_ldr_pil: Image.Image,
    explicit_mask_pil: Image.Image | None,
    masked_foreground_pil: Image.Image | None,
    local_relative_pil: Image.Image | None,
    gaussian_blur_pil: Image.Image | None,
    quantile_estimation_pil: Image.Image | None,
    quantile_highlight_mask_pil: Image.Image | None,
    foreground_quantile_extract_pil: Image.Image | None,
    probe_input_latent_vis_pil: Image.Image | None,
    probe_target_score_pil: Image.Image | None,
    probe_target_mask_pil: Image.Image | None,
    final_pred_pil: Image.Image,
    relighting_abs_diff_pil: Image.Image | None,
    relighting_signed_delta_pil: Image.Image | None,
    input_vs_pred_pil: Image.Image | None,
):
    stage_root = os.path.join(
        save_dir,
        input_image_name,
        "stage_outputs",
        f"{target_lighting_name}_{int(target_view_idx):03d}",
    )
    os.makedirs(stage_root, exist_ok=True)

    ordered_outputs = [
        ("00_input_image.png", input_image_pil),
        ("01_target_envmap_hdr.png", target_envmap_hdr_pil),
        ("02_target_envmap_ldr.png", target_envmap_ldr_pil),
        ("03_explicit_foreground_mask.png", explicit_mask_pil),
        ("04_explicit_mask_extracted_image.png", masked_foreground_pil),
        ("05_local_relative_brightness.png", local_relative_pil),
        ("06_gaussian_blur_image.png", gaussian_blur_pil),
        ("07_quantile_estimation.png", quantile_estimation_pil),
        ("08_quantile_highlight_mask.png", quantile_highlight_mask_pil),
        ("09_foreground_quantile_highlight_extract.png", foreground_quantile_extract_pil),
        ("10_probe_input_latent_visualization.png", probe_input_latent_vis_pil),
        ("11_probe_target_score.png", probe_target_score_pil),
        ("12_probe_target_mask.png", probe_target_mask_pil),
        ("13_final_predicted_image.png", final_pred_pil),
        ("14_relighting_abs_diff_heatmap.png", relighting_abs_diff_pil),
        ("15_relighting_signed_brightness_delta.png", relighting_signed_delta_pil),
        ("16_input_vs_prediction_side_by_side.png", input_vs_pred_pil),
    ]
    written_files = []
    for filename, image in ordered_outputs:
        if image is None:
            continue
        image.save(os.path.join(stage_root, filename))
        written_files.append(filename)

    with open(os.path.join(stage_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_image_name": input_image_name,
                "target_lighting_name": target_lighting_name,
                "target_view_idx": int(target_view_idx),
                "files": written_files,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")


def log_validation(validation_dataloader, vae, image_encoder, feature_extractor, unet, args, accelerator, weight_dtype, img_per_object=600, split="val"):
    """
    对真实图片做 relighting 推理，并把结果保存到磁盘。

    尽管函数名叫 log_validation，这里做的事情本质上是“批量推理”而不是训练中的验证。
    主要流程:
    1. 组装 pipeline
    2. 遍历 dataloader，逐批生成预测图
    3. 把输入图 / 环境图 / 预测图按目录结构保存
    """
    logger.info("Running {} validation... ".format(split))

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = Neural_Gaffer_StableDiffusionPipeline(
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention and accelerator.device.type == "cuda":
        pipeline.enable_xformers_memory_efficient_attention()


    predicted_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    gt_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    LDR_target_environment_maps = []
    HDR_target_environment_maps = []
    input_images = []
    input_image_names = []
    target_lighting_names = []
    target_view_idx_list = []
    stage_dump_count = 0

    for valid_step, batch in tqdm(enumerate(validation_dataloader)):
        if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
            break
        # 真实图推理时没有 GT，因此 batch 里主要是:
        # - image_cond: 输入图
        # - envir_map_target_ldr / hdr: 目标环境光
        # - cond_img_name / target_view_idx / target_envir_map_name: 用于保存文件名
        input_image = batch["image_cond"].to(dtype=weight_dtype)

        target_envmap_ldr = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
        target_envmap_hdr = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
        cond_image_names =  batch["cond_img_name"]
        target_view_idx = batch["target_view_idx"]
        target_envir_map_names = batch["target_envir_map_name"]
        for i in range(len(cond_image_names)):
            input_image_names.append(cond_image_names[i])
            target_lighting_names.append(target_envir_map_names[i])
            target_view_idx_list.append(target_view_idx[i])

        cur_predicted_images = []  # Initialize here
        batchsize, _, h, w = input_image.shape
        cached_stage_output = None
        
        generartor_list = [torch.Generator(device=accelerator.device).manual_seed(args.seed) for _ in range(batchsize)]
        autocast_context = (
            torch.autocast("cuda")
            if accelerator.device.type == "cuda"
            else contextlib.nullcontext()
        )
        for _ in range(args.num_validation_images): # sampled times
            with autocast_context:
                # 真实图推理不传 pose，直接用条件图 + 目标环境图进行 relighting。
                if (
                    getattr(args, "dump_stage_outputs", False)
                    and cached_stage_output is None
                    and int(getattr(args, "dump_stage_max_samples", 0) or 0) > stage_dump_count
                    and int(args.num_validation_images) == 1
                ):
                    cached_stage_output = pipeline(
                        input_imgs=input_image,
                        prompt_imgs=input_image,
                        first_target_envir_map=target_envmap_hdr,
                        second_target_envir_map=target_envmap_ldr,
                        poses=None,
                        height=h,
                        width=w,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=50,
                        generator=generartor_list,
                        return_intermediates=True,
                    )
                    pipeline_output_images = cached_stage_output["images"]
                else:
                    pipeline_output_images = pipeline(input_imgs=input_image, prompt_imgs=input_image, 
                                    first_target_envir_map=target_envmap_hdr, second_target_envir_map=target_envmap_ldr, poses=None, 
                                    height=h, width=w,
                                    guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generartor_list).images

            cur_predicted_images.append(pipeline_output_images) # PIL image list [num_validation_images, batch_size]
            
        
        # [-1, 1][batch_size, 3, h, w] -> [0, 1][batch_size, h, w, 3]
        envir_map_target_hdr_npy = 0.5 * (np.array(target_envmap_hdr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        envir_map_target_ldr_npy = 0.5 * (np.array(target_envmap_ldr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)


        input_image_npy = 0.5 * (np.array(input_image.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        input_images.append(input_image_npy)


        prediction_image_sample0_list = []
        for i in range(batchsize):
            prediction_image_sample0_list.append(np.array(cur_predicted_images[0][i]))
        prediction_image_sample0 = np.array(prediction_image_sample0_list, dtype=np.float32) / 255.0
        # prediction_image_sample1 = np.array([cur_predicted_images[1][i] for i in range(batchsize)], dtype=np.float32) / 255.0
        predicted_images.append(prediction_image_sample0)
        LDR_target_environment_maps.append(envir_map_target_ldr_npy)        
        HDR_target_environment_maps.append(envir_map_target_hdr_npy)

        if getattr(args, "dump_stage_outputs", False) and stage_dump_count < int(getattr(args, "dump_stage_max_samples", 0) or 0):
            if cached_stage_output is None:
                stage_seed_base = int(args.seed) + valid_step * 1000
                stage_generator_list = [
                    torch.Generator(device=accelerator.device).manual_seed(stage_seed_base + i)
                    for i in range(batchsize)
                ]
                stage_autocast_context = (
                    torch.autocast("cuda")
                    if accelerator.device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with stage_autocast_context:
                    stage_output = pipeline(
                        input_imgs=input_image,
                        prompt_imgs=input_image,
                        first_target_envir_map=target_envmap_hdr,
                        second_target_envir_map=target_envmap_ldr,
                        poses=None,
                        height=h,
                        width=w,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=50,
                        generator=stage_generator_list,
                        return_intermediates=True,
                    )
            else:
                stage_output = cached_stage_output
            pred_x0_images = stage_output["intermediates"].get("pred_x0_images")
            pred_x0_latents = stage_output["intermediates"].get("pred_x0_latents")
            final_stage_images = stage_output["images"]
            for i in range(batchsize):
                if stage_dump_count >= int(getattr(args, "dump_stage_max_samples", 0) or 0):
                    break
                pred_x0_pil = pred_x0_images[i] if pred_x0_images is not None else None
                diagnostics = None
                input_tensor = input_image[i : i + 1].detach().cpu()
                final_pred_tensor = transforms.ToTensor()(final_stage_images[i]).unsqueeze(0) * 2.0 - 1.0
                if pred_x0_pil is not None:
                    pred_x0_tensor = transforms.ToTensor()(pred_x0_pil).unsqueeze(0) * 2.0 - 1.0
                    highlight_relative_mode = getattr(args, "highlight_relative_mode", "difference")
                    diagnostics = compute_highlight_diagnostics_for_inference(
                        pred_x0_tensor,
                        background_threshold=getattr(args, "foreground_background_threshold", 0.98),
                        relative_mode=highlight_relative_mode,
                        local_kernel_size=getattr(args, "highlight_local_kernel_size", 15),
                        relative_eps=getattr(args, "highlight_relative_eps", 1e-4),
                        use_quantile_threshold=getattr(args, "highlight_use_quantile_threshold", True),
                        highlight_quantile=getattr(args, "highlight_quantile", 0.88),
                        min_threshold=getattr(args, "highlight_min_threshold", 0.02),
                        max_threshold=getattr(args, "highlight_max_threshold", 0.2),
                        quantile_blur_sigma=getattr(args, "highlight_quantile_blur_sigma", 1.0),
                    )
                save_stage_outputs(
                    save_dir=args.save_dir,
                    input_image_name=cond_image_names[i],
                    target_lighting_name=target_envir_map_names[i],
                    target_view_idx=int(target_view_idx[i]),
                    input_image_pil=Image.fromarray((input_image_npy[i] * 255).astype(np.uint8)),
                    target_envmap_hdr_pil=Image.fromarray((envir_map_target_hdr_npy[i] * 255).astype(np.uint8)),
                    target_envmap_ldr_pil=Image.fromarray((envir_map_target_ldr_npy[i] * 255).astype(np.uint8)),
                    explicit_mask_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["explicit_mask_rgb"])[0]
                    ) if diagnostics is not None else None,
                    masked_foreground_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["masked_foreground_rgb"])[0]
                    ) if diagnostics is not None else None,
                    local_relative_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["local_relative_rgb"])[0]
                    ) if diagnostics is not None else None,
                    gaussian_blur_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["gaussian_blur_rgb"])[0]
                    ) if diagnostics is not None else None,
                    quantile_estimation_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["quantile_estimation_rgb"])[0]
                    ) if diagnostics is not None else None,
                    quantile_highlight_mask_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["quantile_highlight_mask_rgb"])[0]
                    ) if diagnostics is not None else None,
                    foreground_quantile_extract_pil=ensure_pil_image(
                        tensor_bchw_to_numpy_bhwc_01(diagnostics["foreground_quantile_highlight_extract_rgb"])[0]
                    ) if diagnostics is not None else None,
                    probe_input_latent_vis_pil=single_channel_to_rgb_pil(
                        pred_x0_latents[i : i + 1].abs().mean(dim=1, keepdim=True),
                        output_hw=(h, w),
                    ) if pred_x0_latents is not None else None,
                    probe_target_score_pil=single_channel_to_rgb_pil(
                        diagnostics["quantile_score_rgb"][i : i + 1, :1],
                        output_hw=(h, w),
                    ) if diagnostics is not None else None,
                    probe_target_mask_pil=single_channel_to_rgb_pil(
                        diagnostics["quantile_highlight_mask_rgb"][i : i + 1, :1],
                        output_hw=(h, w),
                    ) if diagnostics is not None else None,
                    final_pred_pil=final_stage_images[i],
                    relighting_abs_diff_pil=make_abs_diff_heatmap_pil(
                        input_tensor,
                        final_pred_tensor,
                        diagnostics["explicit_mask_rgb"][i : i + 1, :1] if diagnostics is not None else None,
                    ),
                    relighting_signed_delta_pil=make_signed_brightness_delta_pil(
                        input_tensor,
                        final_pred_tensor,
                        diagnostics["explicit_mask_rgb"][i : i + 1, :1] if diagnostics is not None else None,
                    ),
                    input_vs_pred_pil=make_side_by_side_pil(
                        Image.fromarray((input_image_npy[i] * 255).astype(np.uint8)),
                        final_stage_images[i],
                    ),
                )
                stage_dump_count += 1
            
    predicted_images = np.concatenate(predicted_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    LDR_target_environment_maps = np.concatenate(LDR_target_environment_maps, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    input_images = np.concatenate(input_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    
    
    # 保存结果时按“输入图名字 / 图片类型 / 光照名字_视角编号”的结构落盘，
    # 方便后处理、做视频或者复查单个样本。
    save_dir = os.path.join(args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for image_idx in range(predicted_images.shape[0]):

        cur_predicted_image = predicted_images[image_idx]
        cur_target_envmap_ldr = LDR_target_environment_maps[image_idx]
        cur_input_image = input_images[image_idx]
        cur_input_image_name = input_image_names[image_idx]
        cur_target_lighting_name = target_lighting_names[image_idx]
        cur_target_view_idx = target_view_idx_list[image_idx]
        os.makedirs(f'{save_dir}/{cur_input_image_name}', exist_ok=True)
        os.makedirs(f'{save_dir}/{cur_input_image_name}/input_image', exist_ok=True)
        os.makedirs(f'{save_dir}/{cur_input_image_name}/target_envmap_hdr', exist_ok=True)
        os.makedirs(f'{save_dir}/{cur_input_image_name}/target_envmap_ldr', exist_ok=True)
        os.makedirs(f'{save_dir}/{cur_input_image_name}/pred_image', exist_ok=True)
        input_image_PIL = Image.fromarray((np.squeeze(cur_input_image) * 255).astype(np.uint8))
        cur_target_envmap_hdr = HDR_target_environment_maps[image_idx]
        target_envmap_hdr_PIL = Image.fromarray((np.squeeze(cur_target_envmap_hdr) * 255).astype(np.uint8))
        target_envmap_ldr_PIL = Image.fromarray((np.squeeze(cur_target_envmap_ldr) * 255).astype(np.uint8))
        pred_image_PIL = Image.fromarray((np.squeeze(cur_predicted_image) * 255).astype(np.uint8))

        target_envmap_hdr_PIL.save(f'{save_dir}/{cur_input_image_name}/target_envmap_hdr/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')
        target_envmap_ldr_PIL.save(f'{save_dir}/{cur_input_image_name}/target_envmap_ldr/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')
        pred_image_PIL.save(f'{save_dir}/{cur_input_image_name}/pred_image/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')
        input_image_PIL.save(f'{save_dir}/{cur_input_image_name}/input_image/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')


    return True


def main(args):
    """
    真实图片推理入口。

    这里不会训练模型，只会:
    - 加载 checkpoint
    - 构建真实图数据集
    - 调用 log_validation 批量保存结果
    """
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)



    # Load scheduler and models
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    # 这里把原始 UNet 的输入通道从 8 改到 16。
    # 原因是本项目的 UNet 输入不是“噪声 + 一张条件图”这么简单，
    # 而是要额外拼接 HDR / LDR 目标环境图 latent，因此通道数翻倍。
    conv_in_16 = torch.nn.Conv2d(16, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
    conv_in_16.requires_grad_(False)
    unet.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_16.weight)
    conv_in_16.weight[:,:8,:,:].copy_(unet.conv_in.weight)
    conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    unet.requires_grad_(False)



    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. 'Please make sure to always have all model weights in full float32 precision when starting training'"
        )


    # 构建真实图片推理数据集。
    # 这里的数据已经是预处理后的真实图和目标环境图。
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution), antialias=True),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
 
    

    lighting_per_view = args.lighting_per_view
    total_view = args.total_view
    img_per_object = lighting_per_view * total_view
    validation_dataset = Relighting_Data(
        lighting_dir = args.val_lighting_dir,
        img_dir = args.val_img_dir,   
        lighting_per_view=lighting_per_view,
        total_view=total_view,
        image_transforms=image_transforms,
        specific_object=args.specific_object 

    ) 
    # import ipdb; ipdb.set_trace()
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=4,
        num_workers=0,
        pin_memory=accelerator.device.type == "cuda",
    )

    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)
    
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
    unet.to(accelerator.device, dtype=weight_dtype)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist !!!"
            )
            os._exit(1)
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))

    else:
        print("No checkpoint found. Validation Failed")
        
    print("Loading checkpoint finished!!!!")

    
    if validation_dataloader is not None:

        _ = log_validation(
            validation_dataloader,
            vae,
            image_encoder,
            feature_extractor,
            unet,
            args,
            accelerator,
            weight_dtype,
            split='real_img',
            img_per_object=img_per_object
        )
  
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
