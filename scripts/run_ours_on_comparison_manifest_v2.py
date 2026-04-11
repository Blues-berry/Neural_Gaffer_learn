import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from safetensors.torch import load_model
from torchvision import transforms

try:
    import kornia.geometry.transform as kornia_transform
    from kornia.geometry import pyrup as kornia_pyrup

    if not hasattr(kornia_transform, "build_laplacian_pyramid"):
        def _build_laplacian_pyramid(input_tensor: torch.Tensor, max_level: int):
            if max_level <= 0:
                raise ValueError(f"max_level must be positive, got {max_level}")

            gaussian_pyramid = [input_tensor]
            current = input_tensor
            for _ in range(max_level - 1):
                current = kornia_transform.pyrdown(current)
                gaussian_pyramid.append(current)

            laplacian_pyramid = []
            for level in range(len(gaussian_pyramid) - 1):
                expanded = kornia_pyrup(gaussian_pyramid[level + 1])
                if expanded.shape[-2:] != gaussian_pyramid[level].shape[-2:]:
                    expanded = torch.nn.functional.interpolate(
                        expanded,
                        size=gaussian_pyramid[level].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                laplacian_pyramid.append(gaussian_pyramid[level] - expanded)
            laplacian_pyramid.append(gaussian_pyramid[-1])
            return laplacian_pyramid

        kornia_transform.build_laplacian_pyramid = _build_laplacian_pyramid
except Exception:
    pass

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_relighting_training import NeuralGafferTrainingData
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a local Neural Gaffer checkpoint on a comparison manifest with training-aligned inference defaults."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", default="model_weights/neural_gaffer_model_cache/jbhdfvfc_ckpt80k__neural_gaffer_training_gpu1_highlight")
    parser.add_argument("--pretrained-model-name-or-path", default="kxic/zero123-xl")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint-path", default=None, help="Optional accelerate checkpoint containing model.safetensors.")
    parser.add_argument("--method-name", default="ours_v2")
    parser.add_argument("--output-root", required=True, help="Method root expected by export_relighting_comparison_assets.py")
    parser.add_argument("--device", default="auto" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=256, help="Fallback fixed resolution when --keep-input-resolution is disabled.")
    parser.add_argument("--keep-input-resolution", action="store_true", default=False)
    parser.add_argument("--no-keep-input-resolution", dest="keep_input_resolution", action="store_false")
    parser.add_argument("--max-resolution", type=int, default=512)
    parser.add_argument("--round-to-multiple", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--min-free-memory-mib", type=int, default=4096, help="Auto device mode only: warn/fallback if no CUDA device has at least this much free memory.")
    parser.add_argument("--skip-existing-predictions", action="store_true")

    # Record the same highlight/foreground alignment knobs used by the training-time pipeline.
    parser.add_argument("--foreground-background-threshold", type=float, default=0.96)
    parser.add_argument("--highlight-threshold", type=float, default=0.8)
    parser.add_argument("--highlight-use-quantile-threshold", action="store_true", default=True)
    parser.add_argument("--no-highlight-use-quantile-threshold", dest="highlight_use_quantile_threshold", action="store_false")
    parser.add_argument("--highlight-quantile", type=float, default=0.88)
    parser.add_argument("--highlight-min-threshold", type=float, default=0.02)
    parser.add_argument("--highlight-max-threshold", type=float, default=0.2)
    parser.add_argument("--highlight-quantile-blur-sigma", type=float, default=1.0)
    parser.add_argument("--highlight-relative-mode", type=str, default="difference", choices=["none", "difference", "ratio"])
    parser.add_argument("--highlight-local-kernel-size", type=int, default=15)
    parser.add_argument("--highlight-relative-eps", type=float, default=1e-4)
    return parser.parse_args()


def resolve_device(device_arg: str, min_free_memory_mib: int):
    requested = str(device_arg).strip().lower()
    if requested not in {"auto", "cuda", "cpu"} and not requested.startswith("cuda:"):
        return device_arg
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"

    if requested.startswith("cuda:"):
        return requested
    if requested == "cuda":
        return "cuda"

    best_index = None
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_free = free_bytes
            best_index = index
    if best_index is None:
        return "cpu"

    best_free_mib = int(best_free / (1024 * 1024))
    if best_free_mib < max(int(min_free_memory_mib), 0):
        print(
            f"[resolve_device] best cuda:{best_index} has only {best_free_mib} MiB free; falling back to cpu",
            flush=True,
        )
        return "cpu"

    resolved = f"cuda:{best_index}"
    print(f"[resolve_device] selected {resolved} with {best_free_mib} MiB free", flush=True)
    return resolved


def build_transforms(target_size: tuple[int, int]):
    target_h, target_w = target_size
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((target_h, target_w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def process_pil(image: Image.Image, image_transforms):
    image = image.convert("RGB")
    return image_transforms(image)


def compute_relative_pose(object_dir: Path, view_idx: int):
    rt = np.load(object_dir / f"{view_idx:03d}_RT.npy")
    helper = object.__new__(NeuralGafferTrainingData)
    d_t, _, _ = NeuralGafferTrainingData.get_T(helper, rt, rt)
    return d_t


def expand_unet_conv_in(unet: UNet2DConditionModel):
    conv_in_16 = torch.nn.Conv2d(
        16,
        unet.conv_in.out_channels,
        kernel_size=unet.conv_in.kernel_size,
        padding=unet.conv_in.padding,
    )
    with torch.no_grad():
        torch.nn.init.zeros_(conv_in_16.weight)
        conv_in_16.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    unet.config.in_channels = 16
    return unet


def ensure_unet_input_channels(unet: UNet2DConditionModel):
    if getattr(unet.config, "in_channels", None) == 16:
        return unet
    if getattr(unet, "conv_in", None) is not None and getattr(unet.conv_in, "in_channels", None) == 16:
        unet.config.in_channels = 16
        return unet
    return expand_unet_conv_in(unet)


def resolve_unet_weights(model_dir: Path, checkpoint_path: str | None):
    if checkpoint_path:
        return Path(checkpoint_path)

    exported_unet = model_dir / "unet" / "diffusion_pytorch_model.safetensors"
    if exported_unet.exists():
        return exported_unet

    checkpoints = sorted(
        [p for p in model_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda path: int(path.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No UNet weights found under {model_dir}")
    return checkpoints[-1] / "model.safetensors"


def load_pipeline(
    model_dir: Path,
    pretrained_model_name_or_path: str,
    revision: str | None,
    checkpoint_path: str | None,
    device: str,
    enable_xformers: bool,
):
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"[load_pipeline] device={device} dtype={torch_dtype}", flush=True)

    local_unet_config = model_dir / "unet" / "config.json"
    if checkpoint_path is None and local_unet_config.exists():
        local_config_data = json.loads(local_unet_config.read_text(encoding="utf-8"))
        if local_config_data.get("in_channels") == 16:
            print("[load_pipeline] using local exported pipeline", flush=True)
            scheduler = DDIMScheduler.from_pretrained(str(model_dir), subfolder="scheduler")
            pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
                str(model_dir),
                scheduler=scheduler,
                safety_checker=None,
                torch_dtype=torch_dtype,
            )
            pipeline = pipeline.to(device)
            pipeline.set_progress_bar_config(disable=True)
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            if enable_xformers and device.startswith("cuda"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    print("[load_pipeline] xformers unavailable", flush=True)
            return pipeline, torch_dtype

    scheduler_source = model_dir if (model_dir / "scheduler").exists() else Path(pretrained_model_name_or_path)
    scheduler = DDIMScheduler.from_pretrained(str(scheduler_source), subfolder="scheduler")

    image_encoder_source = model_dir if (model_dir / "image_encoder").exists() else pretrained_model_name_or_path
    vae_source = model_dir if (model_dir / "vae").exists() else pretrained_model_name_or_path
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(image_encoder_source),
        subfolder="image_encoder",
        revision=revision,
    )
    vae = AutoencoderKL.from_pretrained(
        str(vae_source),
        subfolder="vae",
        revision=revision,
    )

    unet_config_dir = None
    if local_unet_config.exists():
        unet_config_dir = model_dir / "unet"
    elif Path(pretrained_model_name_or_path).exists():
        unet_config_dir = Path(pretrained_model_name_or_path) / "unet"
    else:
        candidate = Path("/4T/huggingface_cache/models--kxic--zero123-xl/snapshots/7d8aec2223b93e84eb26893d1e732e013523474b/unet")
        if candidate.exists() and pretrained_model_name_or_path == "kxic/zero123-xl":
            unet_config_dir = candidate
    if unet_config_dir is None or not (unet_config_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Could not resolve local base UNet config for {pretrained_model_name_or_path}. "
            "Pass a local path via --pretrained-model-name-or-path."
        )

    unet_config = json.loads((unet_config_dir / "config.json").read_text(encoding="utf-8"))
    unet = UNet2DConditionModel.from_config(unet_config)
    unet = ensure_unet_input_channels(unet)

    weights_path = resolve_unet_weights(model_dir, checkpoint_path)
    missing_keys, unexpected_keys = load_model(unet, str(weights_path), strict=False, device="cpu")
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"UNet weights mismatch for {weights_path}. Missing={missing_keys[:8]} Unexpected={unexpected_keys[:8]}"
        )

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
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if enable_xformers and device.startswith("cuda"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            print("[load_pipeline] xformers unavailable", flush=True)
    return pipeline, torch_dtype


def sample_key(sample: dict):
    return (
        f"{sample.get('preset', 'na')}_{sample['object_id']}"
        f"_v{int(sample['view_idx']):03d}"
        f"_t{int(sample['target_lighting_index']):03d}"
    )


def resolve_target_size(input_size: tuple[int, int], args) -> tuple[int, int]:
    if args.keep_input_resolution:
        width, height = input_size
    else:
        width = int(args.resolution)
        height = int(args.resolution)

    max_resolution = max(int(args.max_resolution or 0), 0)
    if max_resolution > 0:
        max_side = max(width, height)
        if max_side > max_resolution:
            scale = max_resolution / float(max_side)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))

    round_to = max(int(args.round_to_multiple or 1), 1)
    width = max(round_to, int(round(width / round_to) * round_to))
    height = max(round_to, int(round(height / round_to) * round_to))
    return height, width


def save_prediction(output_root: Path, sample: dict, image: Image.Image):
    object_dir = output_root / sample["object_id"] / "pred_image"
    object_dir.mkdir(parents=True, exist_ok=True)
    object_path = object_dir / sample["target_file"]
    image.save(object_path)

    by_sample_dir = output_root / "_by_sample" / sample_key(sample) / "pred_image"
    by_sample_dir.mkdir(parents=True, exist_ok=True)
    by_sample_path = by_sample_dir / sample["target_file"]
    image.save(by_sample_path)
    return object_path, by_sample_path


def main():
    args = parse_args()
    resolved_device = resolve_device(args.device, args.min_free_memory_mib)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline, weight_dtype = load_pipeline(
        model_dir=Path(args.model_dir),
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        checkpoint_path=args.checkpoint_path,
        device=resolved_device,
        enable_xformers=args.enable_xformers,
    )

    results = []
    for sample in manifest["samples"]:
        existing_object_path = output_root / sample["object_id"] / "pred_image" / sample["target_file"]
        existing_by_sample_path = output_root / "_by_sample" / sample_key(sample) / "pred_image" / sample["target_file"]
        if args.skip_existing_predictions and existing_object_path.exists() and existing_by_sample_path.exists():
            print(
                f"[inference_v2] skip-existing object_id={sample['object_id']} view_idx={sample['view_idx']} target={sample['target_file']}",
                flush=True,
            )
            results.append(
                {
                    "sample_key": sample_key(sample),
                    "object_id": sample["object_id"],
                    "target_file": sample["target_file"],
                    "prediction": str(existing_object_path),
                    "sample_specific_prediction": str(existing_by_sample_path),
                    "prediction_size": None,
                    "target_size_hw": None,
                    "input_size_wh": None,
                    "skipped_existing": True,
                }
            )
            continue
        print(
            f"[inference_v2] object_id={sample['object_id']} view_idx={sample['view_idx']} target={sample['target_file']}",
            flush=True,
        )
        object_dir = Path(sample["gt_path"]).parent
        input_pil = Image.open(sample["input_path"]).convert("RGB")
        target_size = resolve_target_size(input_pil.size, args)
        image_transforms = build_transforms(target_size)

        input_image = process_pil(input_pil, image_transforms).unsqueeze(0).to(resolved_device, dtype=weight_dtype)
        target_env_ldr = process_pil(Image.open(sample["target_lighting_ldr_path"]), image_transforms).unsqueeze(0).to(resolved_device, dtype=weight_dtype)
        target_env_hdr = process_pil(Image.open(sample["target_lighting_hdr_path"]), image_transforms).unsqueeze(0).to(resolved_device, dtype=weight_dtype)
        pose = compute_relative_pose(object_dir, int(sample["view_idx"])).unsqueeze(0).to(resolved_device, dtype=weight_dtype)
        generator_device = resolved_device if resolved_device.startswith("cuda") else "cpu"
        generator_list = [
            torch.Generator(device=generator_device).manual_seed(args.seed + batch_index)
            for batch_index in range(int(input_image.shape[0]))
        ]

        autocast_ctx = torch.autocast("cuda") if resolved_device.startswith("cuda") else contextlib.nullcontext()
        with torch.no_grad():
            with autocast_ctx:
                pred = pipeline(
                    input_imgs=input_image,
                    prompt_imgs=input_image,
                    poses=pose,
                    first_target_envir_map=target_env_hdr,
                    second_target_envir_map=target_env_ldr,
                    height=target_size[0],
                    width=target_size[1],
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator_list,
                ).images[0]

        object_path, by_sample_path = save_prediction(output_root, sample, pred)
        results.append(
            {
                "sample_key": sample_key(sample),
                "object_id": sample["object_id"],
                "target_file": sample["target_file"],
                "prediction": str(object_path),
                "sample_specific_prediction": str(by_sample_path),
                "prediction_size": list(pred.size),
                "target_size_hw": [int(target_size[0]), int(target_size[1])],
                "input_size_wh": [int(input_pil.size[0]), int(input_pil.size[1])],
            }
        )
        print(f"saved {object_path}", flush=True)
        if resolved_device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary = {
        "manifest": args.manifest,
        "output_root": str(output_root),
        "method_name": args.method_name,
        "model_dir": args.model_dir,
        "checkpoint_path": args.checkpoint_path,
        "device": resolved_device,
        "requested_device": args.device,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "keep_input_resolution": args.keep_input_resolution,
        "effective_resolution_policy": "input_native" if args.keep_input_resolution else f"fixed_{int(args.resolution)}",
        "max_resolution": args.max_resolution,
        "round_to_multiple": args.round_to_multiple,
        "generator_mode": "official_style_generator_list",
        "skip_existing_predictions": args.skip_existing_predictions,
        "highlight_alignment_config": {
            "foreground_background_threshold": args.foreground_background_threshold,
            "highlight_threshold": args.highlight_threshold,
            "highlight_use_quantile_threshold": args.highlight_use_quantile_threshold,
            "highlight_quantile": args.highlight_quantile,
            "highlight_min_threshold": args.highlight_min_threshold,
            "highlight_max_threshold": args.highlight_max_threshold,
            "highlight_quantile_blur_sigma": args.highlight_quantile_blur_sigma,
            "highlight_relative_mode": args.highlight_relative_mode,
            "highlight_local_kernel_size": args.highlight_local_kernel_size,
            "highlight_relative_eps": args.highlight_relative_eps,
        },
        "results": results,
    }
    summary_path = output_root / f"{args.method_name}_manifest_predictions.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
