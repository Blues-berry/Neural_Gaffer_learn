import argparse
import json
import os
import random
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kornia
import numpy as np
import torch
import torch.nn.functional as F

from dataset.foreground_mask_utils import fallback_white_background_mask, load_image_array, resolve_foreground_mask


TARGET_IMAGE_PATTERN = re.compile(r"^(\d{3})_(\d{3})_.+\.png$")
ORIGINAL_ASSETS_ROOT = Path(
    os.environ.get(
        "NEURAL_GAFFER_ORIGINAL_ASSETS_ROOT",
        REPO_ROOT / "external_data" / "neural_gaffer_original",
    )
)

DATASET_REGISTRY = {
    "ecommerce": {
        "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_ecommerce_subset"),
        "lighting_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_ecommerce_subset"),
    },
    "landscape": {
        "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_landscape_subset"),
        "lighting_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_landscape_subset"),
    },
    "three_future": {
        "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_three_future_standalone"),
        "lighting_dir": str(
            ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_three_future_standalone"
        ),
    },
}


def summarize_distribution(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def maybe_blur_luminance_for_threshold(luminance, blur_sigma=0.0):
    sigma = float(blur_sigma or 0.0)
    if sigma <= 0.0:
        return luminance
    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    return kornia.filters.gaussian_blur2d(
        luminance,
        (kernel_size, kernel_size),
        (sigma, sigma),
        border_type="reflect",
    )


def compute_highlight_local_mean(luminance, foreground_mask=None, local_kernel_size=15, eps=1e-6):
    kernel_size = max(int(local_kernel_size or 0), 1)
    if kernel_size <= 1:
        return luminance
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    if foreground_mask is None:
        padded = F.pad(luminance, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)

    foreground_mask = foreground_mask.to(device=luminance.device, dtype=luminance.dtype)
    masked = luminance * foreground_mask
    kernel_area = float(kernel_size * kernel_size)
    local_sum = F.avg_pool2d(masked, kernel_size=kernel_size, stride=1, padding=padding) * kernel_area
    local_weight = F.avg_pool2d(foreground_mask, kernel_size=kernel_size, stride=1, padding=padding) * kernel_area
    safe_mean = local_sum / local_weight.clamp_min(float(eps))
    return torch.where(local_weight > float(eps), safe_mean, luminance)


def resolve_highlight_reference_map(
    luminance,
    foreground_mask=None,
    relative_mode="none",
    local_kernel_size=15,
    relative_eps=1e-4,
):
    mode = str(relative_mode or "none").lower()
    if mode == "none":
        return luminance, luminance

    local_mean = compute_highlight_local_mean(
        luminance,
        foreground_mask=foreground_mask,
        local_kernel_size=local_kernel_size,
        eps=relative_eps,
    )
    if mode == "difference":
        return local_mean, luminance - local_mean
    if mode == "ratio":
        return local_mean, luminance / local_mean.clamp_min(float(relative_eps))
    raise ValueError(f"Unsupported highlight_relative_mode: {relative_mode}")


def resolve_highlight_threshold_map(
    luminance,
    foreground_mask,
    threshold=0.8,
    use_quantile_threshold=False,
    highlight_quantile=0.88,
    min_threshold=0.02,
    max_threshold=0.2,
    quantile_blur_sigma=0.0,
    relative_mode="none",
    local_kernel_size=15,
    relative_eps=1e-4,
):
    mode = str(relative_mode or "none").lower()
    if mode == "none":
        base_threshold = float(np.clip(threshold, 0.0, 0.999))
        min_threshold = float(np.clip(min_threshold, 0.0, 0.999))
        max_threshold = float(np.clip(max_threshold, min_threshold, 0.999))
    else:
        base_threshold = float(threshold)
        min_threshold = float(min_threshold)
        max_threshold = float(max(max_threshold, min_threshold))

    quantile = float(np.clip(highlight_quantile, 0.0, 1.0))
    reference_map, _ = resolve_highlight_reference_map(
        luminance,
        foreground_mask=foreground_mask,
        relative_mode=relative_mode,
        local_kernel_size=local_kernel_size,
        relative_eps=relative_eps,
    )
    threshold_luminance = maybe_blur_luminance_for_threshold(
        luminance,
        blur_sigma=quantile_blur_sigma,
    )
    _, threshold_measure = resolve_highlight_reference_map(
        threshold_luminance,
        foreground_mask=foreground_mask,
        relative_mode=relative_mode,
        local_kernel_size=local_kernel_size,
        relative_eps=relative_eps,
    )

    if use_quantile_threshold:
        base_thresholds = torch.full((luminance.shape[0], 1, 1, 1), base_threshold, dtype=luminance.dtype)
        flat_measure = threshold_measure.reshape(threshold_measure.shape[0], -1)
        flat_foreground = foreground_mask.reshape(foreground_mask.shape[0], -1) > 0.5
        for batch_idx in range(luminance.shape[0]):
            foreground_values = flat_measure[batch_idx][flat_foreground[batch_idx]]
            if foreground_values.numel() == 0:
                continue
            quantile_threshold = torch.quantile(foreground_values, quantile)
            quantile_threshold = quantile_threshold.clamp(min=min_threshold, max=max_threshold)
            base_thresholds[batch_idx] = quantile_threshold.to(dtype=luminance.dtype)
    else:
        base_thresholds = torch.full((luminance.shape[0], 1, 1, 1), base_threshold, dtype=luminance.dtype)
        base_thresholds = base_thresholds.clamp(min=min_threshold, max=max_threshold)

    if mode == "none":
        return base_thresholds
    if mode == "difference":
        threshold_map = reference_map + base_thresholds
    elif mode == "ratio":
        threshold_map = reference_map * base_thresholds
    else:
        raise ValueError(f"Unsupported highlight_relative_mode: {relative_mode}")
    return threshold_map.clamp(min=0.0, max=0.999)


def compute_highlight_mask_from_rgb(rgb_image, foreground_mask, args):
    image_tensor = torch.from_numpy(rgb_image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(foreground_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    luminance = (
        0.299 * image_tensor[:, 0:1]
        + 0.587 * image_tensor[:, 1:2]
        + 0.114 * image_tensor[:, 2:3]
    )
    threshold_map = resolve_highlight_threshold_map(
        luminance,
        foreground_mask=mask_tensor,
        threshold=args.highlight_threshold,
        use_quantile_threshold=args.highlight_use_quantile_threshold,
        highlight_quantile=args.highlight_quantile,
        min_threshold=args.highlight_min_threshold,
        max_threshold=args.highlight_max_threshold,
        quantile_blur_sigma=args.highlight_quantile_blur_sigma,
        relative_mode=args.highlight_relative_mode,
        local_kernel_size=args.highlight_local_kernel_size,
        relative_eps=args.highlight_relative_eps,
    )
    mask = (luminance >= threshold_map).float() * mask_tensor
    return mask.squeeze(0).squeeze(0).cpu().numpy()


def load_object_ids(img_dir):
    img_dir = Path(img_dir)
    training_list_path = img_dir / "training_object_list.json"
    if training_list_path.exists():
        return json.loads(training_list_path.read_text())
    return sorted([path.name for path in img_dir.iterdir() if path.is_dir()])


def collect_target_files(object_dir):
    target_files = []
    for path in sorted(Path(object_dir).glob("*.png")):
        if TARGET_IMAGE_PATTERN.match(path.name):
            target_files.append(path)
    return target_files


def resolve_mask_for_target(object_dir, target_path, background_threshold):
    target_array = load_image_array(str(target_path))
    if target_array is None:
        raise FileNotFoundError(f"Failed to load image: {target_path}")
    match = TARGET_IMAGE_PATTERN.match(target_path.name)
    if match is None:
        raise ValueError(f"Unexpected target filename: {target_path.name}")
    view_idx = int(match.group(1))
    resolved_mask, mask_source = resolve_foreground_mask(
        str(object_dir),
        view_idx=view_idx,
        reference_image_path=str(target_path),
    )
    if resolved_mask is None:
        resolved_mask = fallback_white_background_mask(
            target_array[..., :3],
            background_threshold=background_threshold,
        )
        mask_source = "white_background_fallback"
    target_rgb = target_array[..., :3].astype(np.float32)
    if resolved_mask.shape != target_rgb.shape[:2]:
        mask_tensor = torch.from_numpy(np.asarray(resolved_mask, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=target_rgb.shape[:2],
            mode="nearest",
        )
        resolved_mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    return target_rgb, resolved_mask.astype(np.float32), mask_source, view_idx


def lighting_missing_flags(lighting_dir, object_id, target_filename):
    lighting_root = Path(lighting_dir)
    ldr_exists = (lighting_root / "LDR" / object_id / target_filename).exists()
    hdr_rescaled_exists = (lighting_root / "HDR_rescaled" / object_id / target_filename).exists()
    hdr_exists = (lighting_root / "HDR" / object_id / target_filename).exists()
    return {
        "ldr_missing": 0 if ldr_exists else 1,
        "hdr_rescaled_missing": 0 if hdr_rescaled_exists else 1,
        "hdr_missing": 0 if hdr_exists else 1,
        "any_missing": 0 if (ldr_exists and hdr_rescaled_exists and hdr_exists) else 1,
    }


def analyze_dataset(name, cfg, args):
    object_ids = load_object_ids(cfg["img_dir"])
    rng = random.Random(args.seed)
    if args.sample_objects is not None and len(object_ids) > args.sample_objects:
        object_ids = rng.sample(object_ids, args.sample_objects)

    foreground_area_ratios = []
    background_whiteness = []
    highlight_mask_sparsity = []
    highlight_foreground_ratios = []
    random_lighting_coverage = []
    lighting_missing = {
        "ldr_missing": [],
        "hdr_rescaled_missing": [],
        "hdr_missing": [],
        "any_missing": [],
    }
    mask_source_counts = {}
    sampled_image_count = 0

    for object_id in object_ids:
        object_dir = Path(cfg["img_dir"]) / object_id
        target_files = collect_target_files(object_dir)
        if not target_files:
            continue
        if args.sample_images_per_object is not None and len(target_files) > args.sample_images_per_object:
            target_files = rng.sample(target_files, args.sample_images_per_object)

        random_lighting_files = list(object_dir.glob("random_lighting_*.png"))
        random_lighting_coverage.append(len(random_lighting_files) / max(int(args.total_view), 1))

        for target_path in target_files:
            rgb_image, foreground_mask, mask_source, _ = resolve_mask_for_target(
                object_dir,
                target_path,
                background_threshold=args.foreground_background_threshold,
            )
            mask_source_counts[mask_source] = mask_source_counts.get(mask_source, 0) + 1

            foreground_area_ratio = float(np.mean(foreground_mask))
            foreground_area_ratios.append(foreground_area_ratio)

            background_mask = foreground_mask < 0.5
            if np.any(background_mask):
                whiteness = rgb_image.min(axis=-1)[background_mask]
                background_whiteness.append(float(np.mean(whiteness)))

            highlight_mask = compute_highlight_mask_from_rgb(rgb_image, foreground_mask, args)
            highlight_mask_sparsity.append(float(np.mean(highlight_mask)))
            highlight_foreground_ratios.append(
                float(highlight_mask.sum() / max(foreground_mask.sum(), 1.0))
            )

            missing_flags = lighting_missing_flags(cfg["lighting_dir"], object_id, target_path.name)
            for key, value in missing_flags.items():
                lighting_missing[key].append(value)
            sampled_image_count += 1

    return {
        "dataset": name,
        "img_dir": cfg["img_dir"],
        "lighting_dir": cfg["lighting_dir"],
        "sampled_object_count": len(object_ids),
        "sampled_image_count": sampled_image_count,
        "foreground_area_ratio": summarize_distribution(foreground_area_ratios),
        "background_whiteness": summarize_distribution(background_whiteness),
        "highlight_mask_sparsity": summarize_distribution(highlight_mask_sparsity),
        "highlight_foreground_ratio": summarize_distribution(highlight_foreground_ratios),
        "random_lighting_coverage": summarize_distribution(random_lighting_coverage),
        "missing_lighting_ratio": {
            key: summarize_distribution(values)
            for key, values in lighting_missing.items()
        },
        "mask_source_counts": mask_source_counts,
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Precheck Neural Gaffer subdatasets against the current foreground/highlight pipeline.")
    parser.add_argument("--datasets", nargs="+", default=["ecommerce", "landscape", "three_future"])
    parser.add_argument("--sample_objects", type=int, default=64)
    parser.add_argument("--sample_images_per_object", type=int, default=6)
    parser.add_argument("--total_view", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--foreground_background_threshold", type=float, default=0.96)
    parser.add_argument("--highlight_threshold", type=float, default=0.8)
    parser.add_argument("--highlight_use_quantile_threshold", type=lambda x: str(x).lower() in ("1", "true", "yes", "y", "on"), default=True)
    parser.add_argument("--highlight_quantile", type=float, default=0.88)
    parser.add_argument("--highlight_min_threshold", type=float, default=0.02)
    parser.add_argument("--highlight_max_threshold", type=float, default=0.2)
    parser.add_argument("--highlight_quantile_blur_sigma", type=float, default=1.0)
    parser.add_argument("--highlight_relative_mode", type=str, default="difference", choices=["none", "difference", "ratio"])
    parser.add_argument("--highlight_local_kernel_size", type=int, default=15)
    parser.add_argument("--highlight_relative_eps", type=float, default=1e-4)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    unknown = [dataset_name for dataset_name in args.datasets if dataset_name not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown datasets requested: {unknown}")

    results = {
        "config": {
            "datasets": args.datasets,
            "sample_objects": args.sample_objects,
            "sample_images_per_object": args.sample_images_per_object,
            "total_view": args.total_view,
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
        "datasets": [],
    }

    for dataset_name in args.datasets:
        results["datasets"].append(analyze_dataset(dataset_name, DATASET_REGISTRY[dataset_name], args))

    output = json.dumps(results, indent=2)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n")


if __name__ == "__main__":
    main()
