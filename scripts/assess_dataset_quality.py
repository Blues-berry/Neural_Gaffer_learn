import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

try:
    import kornia
except Exception:
    kornia = None

from dataset.foreground_mask_utils import (
    fallback_white_background_mask,
    load_image_array,
    resolve_foreground_mask,
)


TARGET_IMAGE_PATTERN = re.compile(r"^(\d{3})_(\d{3})_.+\.png$")


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


def normalize_rgb(rgb_image):
    if rgb_image is None:
        return None
    rgb = np.asarray(rgb_image, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def maybe_blur_luminance_for_threshold(luminance, blur_sigma=0.0):
    sigma = float(blur_sigma or 0.0)
    if sigma <= 0.0 or kornia is None:
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
    rgb = normalize_rgb(rgb_image)
    if rgb is None:
        return None
    foreground_mask = np.asarray(foreground_mask, dtype=np.float32)
    if foreground_mask.shape != rgb.shape[:2]:
        mask_tensor = torch.from_numpy(foreground_mask).unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=rgb.shape[:2],
            mode="nearest",
        )
        foreground_mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    image_tensor = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
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
    all_png = list(Path(object_dir).glob("*.png"))
    for path in sorted(all_png):
        if TARGET_IMAGE_PATTERN.match(path.name):
            target_files.append(path)
    return all_png, target_files


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
    target_rgb = normalize_rgb(target_array[..., :3])
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
    hdr_raw_exists = (lighting_root / "HDR_raw" / object_id / target_filename).exists()
    return {
        "ldr_missing": 0 if ldr_exists else 1,
        "hdr_rescaled_missing": 0 if hdr_rescaled_exists else 1,
        "hdr_raw_missing": 0 if hdr_raw_exists else 1,
        "any_missing": 0 if (ldr_exists and hdr_rescaled_exists) else 1,
    }


def analyze_dataset(name, cfg, args):
    object_ids = load_object_ids(cfg["img_dir"])
    total_object_count = len(object_ids)
    rng = random.Random(args.seed)
    if args.sample_objects is not None and len(object_ids) > args.sample_objects:
        object_ids = rng.sample(object_ids, args.sample_objects)

    foreground_area_ratios = []
    background_whiteness = []
    highlight_mask_sparsity = []
    highlight_foreground_ratios = []
    random_lighting_coverage = []
    image_name_match_ratio = []
    lighting_missing = {
        "ldr_missing": [],
        "hdr_rescaled_missing": [],
        "hdr_raw_missing": [],
        "any_missing": [],
    }
    mask_source_counts = {"explicit_mask": 0, "white_background_fallback": 0}
    sampled_image_count = 0

    for object_id in object_ids:
        object_dir = Path(cfg["img_dir"]) / object_id
        all_png, target_files = collect_target_files(object_dir)
        if not all_png:
            continue
        image_name_match_ratio.append(len(target_files) / max(len(all_png), 1))
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
            if mask_source == "white_background_fallback":
                mask_source_counts["white_background_fallback"] += 1
            else:
                mask_source_counts["explicit_mask"] += 1

            foreground_area_ratio = float(np.mean(foreground_mask))
            foreground_area_ratios.append(foreground_area_ratio)

            background_mask = foreground_mask < 0.5
            if np.any(background_mask):
                whiteness = rgb_image.min(axis=-1)[background_mask]
                background_whiteness.append(float(np.mean(whiteness)))

            highlight_mask = compute_highlight_mask_from_rgb(rgb_image, foreground_mask, args)
            if highlight_mask is not None:
                highlight_mask_sparsity.append(float(np.mean(highlight_mask)))
                highlight_foreground_ratios.append(
                    float(highlight_mask.sum() / max(foreground_mask.sum(), 1.0))
                )

            missing_flags = lighting_missing_flags(cfg["lighting_dir"], object_id, target_path.name)
            for key, value in missing_flags.items():
                lighting_missing[key].append(value)
            sampled_image_count += 1

    mask_total = mask_source_counts["explicit_mask"] + mask_source_counts["white_background_fallback"]
    mask_fallback_ratio = None
    if mask_total > 0:
        mask_fallback_ratio = mask_source_counts["white_background_fallback"] / mask_total

    return {
        "dataset": name,
        "img_dir": cfg["img_dir"],
        "lighting_dir": cfg["lighting_dir"],
        "total_object_count": total_object_count,
        "sampled_object_count": len(object_ids),
        "sampled_image_count": sampled_image_count,
        "foreground_area_ratio": summarize_distribution(foreground_area_ratios),
        "background_whiteness": summarize_distribution(background_whiteness),
        "highlight_mask_sparsity": summarize_distribution(highlight_mask_sparsity),
        "highlight_foreground_ratio": summarize_distribution(highlight_foreground_ratios),
        "random_lighting_coverage": summarize_distribution(random_lighting_coverage),
        "image_name_match_ratio": summarize_distribution(image_name_match_ratio),
        "missing_lighting_ratio": {
            key: summarize_distribution(values)
            for key, values in lighting_missing.items()
        },
        "mask_source_counts": mask_source_counts,
        "mask_fallback_ratio": mask_fallback_ratio,
    }


def derive_quality_flags(stats):
    flags = []
    bg_mean = stats["background_whiteness"]["mean"]
    if bg_mean is not None and bg_mean < 0.9:
        flags.append("background_not_white")

    fg_mean = stats["foreground_area_ratio"]["mean"]
    if fg_mean is not None and fg_mean < 0.1:
        flags.append("foreground_small")

    highlight_mean = stats["highlight_mask_sparsity"]["mean"]
    if highlight_mean is not None and highlight_mean < 0.01:
        flags.append("highlight_sparse")

    missing_any = stats["missing_lighting_ratio"]["any_missing"]["mean"]
    if missing_any is not None and missing_any > 0.01:
        flags.append("missing_lighting_files")

    random_cov = stats["random_lighting_coverage"]["mean"]
    if random_cov is not None and random_cov < 0.5:
        flags.append("random_lighting_coverage_low")

    fallback_ratio = stats.get("mask_fallback_ratio")
    if fallback_ratio is not None and fallback_ratio > 0.5:
        flags.append("mask_fallback_high")

    return flags


def render_markdown(results):
    lines = []
    lines.append("# Dataset Quality Report")
    lines.append("")
    lines.append(f"Generated at: {results['generated_at_utc']}")
    lines.append("")
    lines.append("## Config")
    for key, value in results["config"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for stats in results["datasets"]:
        dataset_name = stats["dataset"]
        lines.append(f"## {dataset_name}")
        lines.append(f"- total_object_count: {stats['total_object_count']}")
        lines.append(f"- sampled_object_count: {stats['sampled_object_count']}")
        lines.append(f"- sampled_image_count: {stats['sampled_image_count']}")
        lines.append(f"- mask_fallback_ratio: {stats.get('mask_fallback_ratio')}")
        lines.append("")
        lines.append("Key metrics:")
        lines.append(f"- foreground_area_ratio mean: {stats['foreground_area_ratio']['mean']}")
        lines.append(f"- background_whiteness mean: {stats['background_whiteness']['mean']}")
        lines.append(f"- highlight_mask_sparsity mean: {stats['highlight_mask_sparsity']['mean']}")
        lines.append(f"- highlight_foreground_ratio mean: {stats['highlight_foreground_ratio']['mean']}")
        lines.append(f"- random_lighting_coverage mean: {stats['random_lighting_coverage']['mean']}")
        lines.append(f"- image_name_match_ratio mean: {stats['image_name_match_ratio']['mean']}")
        lines.append(f"- missing_lighting any_missing mean: {stats['missing_lighting_ratio']['any_missing']['mean']}")
        flags = derive_quality_flags(stats)
        if flags:
            lines.append(f"- quality_flags: {', '.join(flags)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_argparser():
    parser = argparse.ArgumentParser(description="Assess dataset quality for Neural Gaffer preprocessed data.")
    parser.add_argument("--ready_root", type=str, default=str(REPO_ROOT / "logs" / "ready_subdatasets_20260328"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--sample_objects", type=int, default=48)
    parser.add_argument("--sample_images_per_object", type=int, default=6)
    parser.add_argument("--total_view", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)

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

    ready_root = Path(args.ready_root)
    if not ready_root.exists():
        raise FileNotFoundError(f"Ready root not found: {ready_root}")

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted([p.name for p in ready_root.iterdir() if p.is_dir() and not p.name.startswith(".")])

    dataset_configs = {}
    for dataset_name in dataset_names:
        dataset_dir = ready_root / dataset_name
        img_dir = dataset_dir / "images"
        lighting_dir = dataset_dir / "lighting"
        if not img_dir.exists() or not lighting_dir.exists():
            raise FileNotFoundError(f"Missing images or lighting for {dataset_name} in {dataset_dir}")
        dataset_configs[dataset_name] = {
            "img_dir": str(img_dir),
            "lighting_dir": str(lighting_dir),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "logs" / f"dataset_quality_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ready_root": str(ready_root),
            "datasets": dataset_names,
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
            "kornia_available": bool(kornia is not None),
        },
        "datasets": [],
    }

    for dataset_name in dataset_names:
        results["datasets"].append(analyze_dataset(dataset_name, dataset_configs[dataset_name], args))

    json_path = output_dir / "dataset_quality.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    md_path = output_dir / "dataset_quality.md"
    md_path.write_text(render_markdown(results))

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote report: {md_path}")


if __name__ == "__main__":
    main()
