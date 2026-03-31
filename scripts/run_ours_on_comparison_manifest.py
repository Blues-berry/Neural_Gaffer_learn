import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from safetensors.torch import load_model
from torchvision import transforms

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline
from dataset.dataset_relighting_training import NeuralGafferTrainingData


def parse_args():
    parser = argparse.ArgumentParser(description="Run our relighting model on a comparison manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", default="logs/neural_gaffer_training_gpu1_highlight")
    parser.add_argument("--pretrained-model-name-or-path", default="kxic/zero123-xl")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint-path", default=None, help="Optional accelerate checkpoint containing model.safetensors.")
    parser.add_argument("--method-name", default="ours")
    parser.add_argument("--output-root", required=True, help="Method root expected by export_relighting_comparison_assets.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-xformers", action="store_true")
    return parser.parse_args()


def build_transforms(resolution: int):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((resolution, resolution), antialias=True),
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
            print("[load_pipeline] local exported pipeline moved to device", flush=True)
            pipeline.set_progress_bar_config(disable=True)
            if enable_xformers and device.startswith("cuda"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("[load_pipeline] xformers enabled", flush=True)
                except Exception:
                    print("[load_pipeline] xformers unavailable", flush=True)
            return pipeline, torch_dtype

    scheduler_source = model_dir if (model_dir / "scheduler").exists() else Path(pretrained_model_name_or_path)
    print(f"[load_pipeline] scheduler_source={scheduler_source}", flush=True)
    scheduler = DDIMScheduler.from_pretrained(str(scheduler_source), subfolder="scheduler")

    image_encoder_source = model_dir if (model_dir / "image_encoder").exists() else pretrained_model_name_or_path
    vae_source = model_dir if (model_dir / "vae").exists() else pretrained_model_name_or_path
    print(f"[load_pipeline] image_encoder_source={image_encoder_source}", flush=True)
    print(f"[load_pipeline] vae_source={vae_source}", flush=True)

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
    print(f"[load_pipeline] base unet instantiated from config {unet_config_dir}", flush=True)
    unet = ensure_unet_input_channels(unet)

    weights_path = resolve_unet_weights(model_dir, checkpoint_path)
    print(f"[load_pipeline] weights_path={weights_path}", flush=True)
    missing_keys, unexpected_keys = load_model(unet, str(weights_path), strict=False, device="cpu")
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"UNet weights mismatch for {weights_path}. Missing={missing_keys[:8]} Unexpected={unexpected_keys[:8]}"
        )
    print("[load_pipeline] unet weights loaded", flush=True)

    pipeline = Neural_Gaffer_StableDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    print("[load_pipeline] pipeline assembled", flush=True)
    pipeline = pipeline.to(device)
    print("[load_pipeline] pipeline moved to device", flush=True)
    pipeline.set_progress_bar_config(disable=True)
    if enable_xformers and device.startswith("cuda"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("[load_pipeline] xformers enabled", flush=True)
        except Exception:
            print("[load_pipeline] xformers unavailable", flush=True)
    return pipeline, torch_dtype


def save_prediction(output_root: Path, object_id: str, target_file: str, image: Image.Image):
    out_dir = output_root / object_id / "pred_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / target_file
    image.save(out_path)
    return out_path


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline, weight_dtype = load_pipeline(
        model_dir=Path(args.model_dir),
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        enable_xformers=args.enable_xformers,
    )
    image_transforms = build_transforms(args.resolution)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    results = []
    for sample in manifest["samples"]:
        print(
            f"[inference] object_id={sample['object_id']} view_idx={sample['view_idx']} target={sample['target_file']}",
            flush=True,
        )
        object_id = sample["object_id"]
        object_dir = Path(sample["gt_path"]).parent

        input_image = process_pil(Image.open(sample["input_path"]), image_transforms).unsqueeze(0).to(args.device, dtype=weight_dtype)
        target_env_ldr = process_pil(Image.open(sample["target_lighting_ldr_path"]), image_transforms).unsqueeze(0).to(args.device, dtype=weight_dtype)
        target_env_hdr_path = Path(sample["target_lighting_hdr_path"])
        target_env_hdr = process_pil(Image.open(target_env_hdr_path), image_transforms).unsqueeze(0).to(args.device, dtype=weight_dtype)
        pose = compute_relative_pose(object_dir, int(sample["view_idx"])).unsqueeze(0).to(args.device, dtype=weight_dtype)

        autocast_ctx = torch.autocast("cuda") if args.device.startswith("cuda") else contextlib.nullcontext()
        with torch.no_grad():
            with autocast_ctx:
                print("[inference] pipeline forward start", flush=True)
                pred = pipeline(
                    input_imgs=input_image,
                    prompt_imgs=input_image,
                    poses=pose,
                    first_target_envir_map=target_env_hdr,
                    second_target_envir_map=target_env_ldr,
                    height=args.resolution,
                    width=args.resolution,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).images[0]
                print("[inference] pipeline forward done", flush=True)

        out_path = save_prediction(output_root, object_id, sample["target_file"], pred)
        results.append({
            "object_id": object_id,
            "target_file": sample["target_file"],
            "prediction": str(out_path),
        })
        print(f"saved {out_path}")

    summary_path = output_root / f"{args.method_name}_manifest_predictions.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
