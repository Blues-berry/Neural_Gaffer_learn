import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.signal
import torch
from PIL import Image

from scripts.assess_dataset_quality import (
    compute_highlight_mask_from_rgb,
    normalize_rgb,
    resolve_mask_for_target,
    summarize_distribution,
)

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    LearnedPerceptualImagePatchSimilarity = None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Evaluate highlight-focused metrics on an exported relighting assets manifest."
    )
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--methods", nargs="*", default=None, help="Explicit method names to evaluate.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--compute-lpips", type=parse_bool, default=True)
    parser.add_argument("--compute-ssim", type=parse_bool, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of manifest samples to evaluate.")
    parser.add_argument("--highlight-crop-padding", type=int, default=12)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    parser.add_argument("--ssim-filter-size", type=int, default=11)
    parser.add_argument("--ssim-filter-sigma", type=float, default=1.5)

    parser.add_argument("--foreground_background_threshold", type=float, default=0.96)
    parser.add_argument("--highlight_threshold", type=float, default=0.8)
    parser.add_argument("--highlight_use_quantile_threshold", type=parse_bool, default=True)
    parser.add_argument("--highlight_quantile", type=float, default=0.88)
    parser.add_argument("--highlight_min_threshold", type=float, default=0.02)
    parser.add_argument("--highlight_max_threshold", type=float, default=0.2)
    parser.add_argument("--highlight_quantile_blur_sigma", type=float, default=1.0)
    parser.add_argument("--highlight_relative_mode", type=str, default="difference", choices=["none", "difference", "ratio"])
    parser.add_argument("--highlight_local_kernel_size", type=int, default=15)
    parser.add_argument("--highlight_relative_eps", type=float, default=1e-4)
    return parser


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / path


def load_rgb(path_value: str):
    path = resolve_repo_path(path_value)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Image not found: {path_value}")
    image = normalize_rgb(np.asarray(Image.open(path)))
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return image


def resize_rgb(rgb: np.ndarray, target_hw: tuple[int, int]):
    target_h, target_w = target_hw
    if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
        return rgb
    image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def mse_to_psnr(mse: float | None):
    if mse is None:
        return None
    mse = max(float(mse), 1e-8)
    return float(10.0 * np.log10(1.0 / mse))


def masked_rgb_mse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    mask = np.asarray(mask, dtype=np.float32)
    denom = float(mask.sum()) * 3.0
    if denom <= 1e-8:
        return None
    diff = (pred - gt) ** 2
    return float((diff * mask[..., None]).sum() / denom)


def masked_rgb_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    mask = np.asarray(mask, dtype=np.float32)
    denom = float(mask.sum()) * 3.0
    if denom <= 1e-8:
        return None
    diff = np.abs(pred - gt)
    return float((diff * mask[..., None]).sum() / denom)


def masked_mean(values: np.ndarray, mask: np.ndarray):
    mask = np.asarray(mask, dtype=np.float32)
    denom = float(mask.sum())
    if denom <= 1e-8:
        return None
    return float((values * mask).sum() / denom)


def compute_luminance(rgb: np.ndarray):
    return (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def bbox_from_mask(mask: np.ndarray, padding: int = 0):
    ys, xs = np.nonzero(mask > 0.5)
    if ys.size == 0 or xs.size == 0:
        return None
    h, w = mask.shape
    top = max(int(ys.min()) - padding, 0)
    bottom = min(int(ys.max()) + padding + 1, h)
    left = max(int(xs.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, w)
    return top, bottom, left, right


def crop_rgb(rgb: np.ndarray, bbox):
    if bbox is None:
        return None
    top, bottom, left, right = bbox
    return rgb[top:bottom, left:right]


def ensure_tensor_image(rgb: np.ndarray, device: str):
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def compute_lpips_distance(model, pred_rgb: np.ndarray, gt_rgb: np.ndarray, device: str):
    if model is None:
        return None
    pred_tensor = ensure_tensor_image(pred_rgb, device) * 2.0 - 1.0
    gt_tensor = ensure_tensor_image(gt_rgb, device) * 2.0 - 1.0
    with torch.inference_mode():
        return float(model(pred_tensor, gt_tensor).mean().item())


def rgb_ssim(img0, img1, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if len(img0.shape) != 3 or img0.shape[-1] != 3 or img0.shape != img1.shape:
        raise ValueError(f"Expected two RGB images of the same shape, got {img0.shape} and {img1.shape}")

    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    filt_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * filt_i)
    filt /= np.sum(filt)

    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    def filt_fn(z):
        return np.stack(
            [
                convolve2d(convolve2d(z[..., channel], filt[:, None]), filt[None, :])
                for channel in range(z.shape[-1])
            ],
            axis=-1,
        )

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    return float(np.mean(numer / denom))


def compute_ssim_value(pred_rgb: np.ndarray, gt_rgb: np.ndarray, filter_size: int = 11, filter_sigma: float = 1.5):
    height, width = pred_rgb.shape[:2]
    max_filter_size = min(int(filter_size), int(height), int(width))
    if max_filter_size < 3:
        return None
    if max_filter_size % 2 == 0:
        max_filter_size -= 1
    if max_filter_size < 3:
        return None
    sigma = min(float(filter_sigma), max(0.5, max_filter_size / 6.0))
    return float(
        rgb_ssim(
            np.asarray(gt_rgb, dtype=np.float32),
            np.asarray(pred_rgb, dtype=np.float32),
            1.0,
            filter_size=max_filter_size,
            filter_sigma=sigma,
        )
    )


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    a = mask_a > 0.5
    b = mask_b > 0.5
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def compute_weighted_centroid(mask: np.ndarray, weights: np.ndarray | None = None):
    binary = mask > 0.5
    if not np.any(binary):
        return None
    ys, xs = np.nonzero(binary)
    if weights is None:
        w = np.ones_like(ys, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)[binary]
    denom = float(w.sum())
    if denom <= 1e-8:
        return None
    return float((ys * w).sum() / denom), float((xs * w).sum() / denom)


def compute_centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray, weights_a: np.ndarray | None = None, weights_b: np.ndarray | None = None):
    centroid_a = compute_weighted_centroid(mask_a, weights=weights_a)
    centroid_b = compute_weighted_centroid(mask_b, weights=weights_b)
    if centroid_a is None or centroid_b is None:
        return None
    h, w = mask_a.shape
    dy = (centroid_a[0] - centroid_b[0]) / max(h - 1, 1)
    dx = (centroid_a[1] - centroid_b[1]) / max(w - 1, 1)
    return float(np.sqrt(dx * dx + dy * dy))


def compute_highlight_chroma(rgb: np.ndarray, mask: np.ndarray):
    binary = mask > 0.5
    if not np.any(binary):
        return None
    pixels = np.asarray(rgb, dtype=np.float32)[binary]
    denom = pixels.sum(axis=1, keepdims=True).clip(1e-6, None)
    chroma = pixels / denom
    return chroma.mean(axis=0)


def chroma_l1(pred_rgb: np.ndarray, gt_rgb: np.ndarray, mask: np.ndarray):
    pred_chroma = compute_highlight_chroma(pred_rgb, mask)
    gt_chroma = compute_highlight_chroma(gt_rgb, mask)
    if pred_chroma is None or gt_chroma is None:
        return None
    return float(np.abs(pred_chroma - gt_chroma).mean())


def compute_sample_metrics(pred_rgb: np.ndarray, gt_rgb: np.ndarray, foreground_mask: np.ndarray, gt_highlight_mask: np.ndarray, pred_highlight_mask: np.ndarray, args):
    non_highlight_mask = np.clip(foreground_mask - gt_highlight_mask, 0.0, 1.0)
    gt_luma = compute_luminance(gt_rgb)
    pred_luma = compute_luminance(pred_rgb)
    union_highlight = np.clip((gt_highlight_mask > 0.5).astype(np.float32) + (pred_highlight_mask > 0.5).astype(np.float32), 0.0, 1.0)

    full_mse = float(np.mean((pred_rgb - gt_rgb) ** 2))
    fg_mse = masked_rgb_mse(pred_rgb, gt_rgb, foreground_mask)
    highlight_mse = masked_rgb_mse(pred_rgb, gt_rgb, gt_highlight_mask)
    non_highlight_mse = masked_rgb_mse(pred_rgb, gt_rgb, non_highlight_mask)
    highlight_mae = masked_rgb_mae(pred_rgb, gt_rgb, gt_highlight_mask)

    gt_highlight_area = masked_mean((gt_highlight_mask > 0.5).astype(np.float32), foreground_mask)
    pred_highlight_area = masked_mean((pred_highlight_mask > 0.5).astype(np.float32), foreground_mask)
    gt_saturated_ratio = masked_mean((gt_rgb.max(axis=-1) >= float(args.saturation_threshold)).astype(np.float32), gt_highlight_mask)
    pred_saturated_ratio = masked_mean((pred_rgb.max(axis=-1) >= float(args.saturation_threshold)).astype(np.float32), gt_highlight_mask)
    gt_p95_luma = float(np.quantile(gt_luma[gt_highlight_mask > 0.5], 0.95)) if np.any(gt_highlight_mask > 0.5) else None
    pred_p95_luma = float(np.quantile(pred_luma[gt_highlight_mask > 0.5], 0.95)) if np.any(gt_highlight_mask > 0.5) else None
    gt_p99_luma = float(np.quantile(gt_luma[gt_highlight_mask > 0.5], 0.99)) if np.any(gt_highlight_mask > 0.5) else None
    pred_p99_luma = float(np.quantile(pred_luma[gt_highlight_mask > 0.5], 0.99)) if np.any(gt_highlight_mask > 0.5) else None

    return {
        "full_mse": full_mse,
        "full_psnr": mse_to_psnr(full_mse),
        "foreground_mse": fg_mse,
        "foreground_psnr": mse_to_psnr(fg_mse),
        "highlight_mse": highlight_mse,
        "highlight_psnr": mse_to_psnr(highlight_mse),
        "highlight_rmse": float(np.sqrt(highlight_mse)) if highlight_mse is not None else None,
        "highlight_mae": highlight_mae,
        "non_highlight_mse": non_highlight_mse,
        "non_highlight_psnr": mse_to_psnr(non_highlight_mse),
        "highlight_mse_ratio": float(highlight_mse / max(non_highlight_mse, 1e-8)) if highlight_mse is not None and non_highlight_mse is not None else None,
        "gt_highlight_area_in_fg": gt_highlight_area,
        "pred_highlight_area_in_fg": pred_highlight_area,
        "highlight_area_abs_error": abs(pred_highlight_area - gt_highlight_area) if pred_highlight_area is not None and gt_highlight_area is not None else None,
        "highlight_mask_iou": compute_mask_iou(pred_highlight_mask, gt_highlight_mask),
        "highlight_centroid_distance": compute_centroid_distance(
            pred_highlight_mask,
            gt_highlight_mask,
            weights_a=pred_luma,
            weights_b=gt_luma,
        ),
        "gt_highlight_saturated_ratio": gt_saturated_ratio,
        "pred_highlight_saturated_ratio": pred_saturated_ratio,
        "highlight_saturated_ratio_abs_error": abs(pred_saturated_ratio - gt_saturated_ratio) if pred_saturated_ratio is not None and gt_saturated_ratio is not None else None,
        "highlight_p95_luma_abs_error": abs(pred_p95_luma - gt_p95_luma) if pred_p95_luma is not None and gt_p95_luma is not None else None,
        "highlight_p99_luma_abs_error": abs(pred_p99_luma - gt_p99_luma) if pred_p99_luma is not None and gt_p99_luma is not None else None,
        "highlight_chroma_l1_on_gt_mask": chroma_l1(pred_rgb, gt_rgb, gt_highlight_mask),
        "union_highlight_coverage_in_fg": masked_mean(union_highlight, foreground_mask),
    }


def find_prediction_path(sample: dict, method_name: str):
    method_entry = sample.get("methods", {}).get(method_name, {})
    candidate_keys = ("source", "composited")
    for key in candidate_keys:
        value = method_entry.get(key)
        if value and resolve_repo_path(value).exists():
            return str(resolve_repo_path(value))
    return None


def infer_methods(samples: list[dict], requested_methods: list[str] | None):
    if requested_methods:
        return requested_methods
    methods = set()
    for sample in samples:
        methods.update(sample.get("methods", {}).keys())
    return sorted(methods)


def init_lpips_if_available(args):
    if not getattr(args, "compute_lpips", True):
        return None, None
    if LearnedPerceptualImagePatchSimilarity is None:
        return None, "torchmetrics LPIPS is unavailable; LPIPS metrics were skipped."
    device = str(args.device)
    try:
        model = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device).eval()
        return model, None
    except Exception as exc:
        return None, f"LPIPS initialization failed on device={device}: {exc}"


def collect_metric_values(records: list[dict], metric_name: str):
    values = []
    for record in records:
        value = record["metrics"].get(metric_name)
        if value is None:
            continue
        if not np.isfinite(value):
            continue
        values.append(float(value))
    return values


def summarize_records(records: list[dict]):
    metric_names = set()
    for record in records:
        metric_names.update(record["metrics"].keys())
    summary = {}
    for metric_name in sorted(metric_names):
        summary[metric_name] = summarize_distribution(collect_metric_values(records, metric_name))
    return summary


def render_markdown(results: dict):
    lines = []
    lines.append("# Highlight Evaluation Summary")
    lines.append("")
    lines.append(f"- generated_at_utc: {results['generated_at_utc']}")
    lines.append(f"- assets_manifest: {results['assets_manifest']}")
    lines.append(f"- evaluated_methods: {', '.join(results['evaluated_methods'])}")
    if results.get("warnings"):
        lines.append(f"- warnings: {' | '.join(results['warnings'])}")
    lines.append("")

    key_metrics = [
        "full_psnr",
        "full_ssim",
        "foreground_psnr",
        "foreground_ssim",
        "highlight_psnr",
        "highlight_rmse",
        "highlight_mse_ratio",
        "highlight_mask_iou",
        "highlight_area_abs_error",
        "highlight_saturated_ratio_abs_error",
        "highlight_p95_luma_abs_error",
        "highlight_chroma_l1_on_gt_mask",
        "highlight_crop_ssim",
        "lpips_full",
        "lpips_foreground",
        "lpips_highlight_crop",
    ]

    for method_name, method_payload in results["methods"].items():
        lines.append(f"## {method_name}")
        lines.append("")
        lines.append("| metric | mean |")
        lines.append("| --- | ---: |")
        overall = method_payload["overall"]
        for metric_name in key_metrics:
            metric_summary = overall.get(metric_name, {})
            mean_value = metric_summary.get("mean")
            if mean_value is None:
                continue
            lines.append(f"| {metric_name} | {mean_value:.6f} |")
        lines.append("")

        by_preset = method_payload.get("by_preset", {})
        if by_preset:
            lines.append("| preset | full_psnr | full_ssim | foreground_psnr | highlight_psnr | highlight_mask_iou |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for preset_name in sorted(by_preset.keys()):
                preset_summary = by_preset[preset_name]
                full_psnr = preset_summary.get("full_psnr", {}).get("mean")
                full_ssim = preset_summary.get("full_ssim", {}).get("mean")
                fg_psnr = preset_summary.get("foreground_psnr", {}).get("mean")
                hl_psnr = preset_summary.get("highlight_psnr", {}).get("mean")
                hl_iou = preset_summary.get("highlight_mask_iou", {}).get("mean")
                lines.append(
                    "| "
                    + preset_name
                    + " | "
                    + (f"{full_psnr:.6f}" if full_psnr is not None else "-")
                    + " | "
                    + (f"{full_ssim:.6f}" if full_ssim is not None else "-")
                    + " | "
                    + (f"{fg_psnr:.6f}" if fg_psnr is not None else "-")
                    + " | "
                    + (f"{hl_psnr:.6f}" if hl_psnr is not None else "-")
                    + " | "
                    + (f"{hl_iou:.6f}" if hl_iou is not None else "-")
                    + " |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = build_argparser()
    args = parser.parse_args()

    assets_manifest_path = resolve_repo_path(args.assets_manifest)
    assets_manifest = json.loads(assets_manifest_path.read_text(encoding="utf-8"))
    samples = assets_manifest.get("samples", [])
    if args.max_samples is not None:
        samples = samples[: max(int(args.max_samples), 0)]
    methods = infer_methods(samples, args.methods)

    lpips_model, lpips_warning = init_lpips_if_available(args)
    warnings = []
    if lpips_warning:
        warnings.append(lpips_warning)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets_manifest": str(assets_manifest_path),
        "evaluated_methods": methods,
        "config": {
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
            "compute_lpips": args.compute_lpips,
            "compute_ssim": args.compute_ssim,
            "device": args.device,
            "max_samples": args.max_samples,
            "highlight_crop_padding": args.highlight_crop_padding,
            "saturation_threshold": args.saturation_threshold,
            "ssim_filter_size": args.ssim_filter_size,
            "ssim_filter_sigma": args.ssim_filter_sigma,
        },
        "warnings": warnings,
        "methods": {},
    }

    method_records = {method_name: [] for method_name in methods}

    for sample in samples:
        gt_path = sample.get("gt_path")
        if not gt_path:
            continue
        gt_path_resolved = resolve_repo_path(gt_path)
        object_dir = gt_path_resolved.parent
        gt_rgb, foreground_mask, mask_source, view_idx = resolve_mask_for_target(
            object_dir,
            gt_path_resolved,
            background_threshold=args.foreground_background_threshold,
        )
        gt_rgb = normalize_rgb(gt_rgb)
        gt_highlight_mask = compute_highlight_mask_from_rgb(gt_rgb, foreground_mask, args)

        for method_name in methods:
            pred_path = find_prediction_path(sample, method_name)
            if pred_path is None:
                continue
            pred_rgb = load_rgb(pred_path)
            pred_rgb = resize_rgb(pred_rgb, gt_rgb.shape[:2])
            pred_highlight_mask = compute_highlight_mask_from_rgb(pred_rgb, foreground_mask, args)

            metrics = compute_sample_metrics(
                pred_rgb=pred_rgb,
                gt_rgb=gt_rgb,
                foreground_mask=foreground_mask,
                gt_highlight_mask=gt_highlight_mask,
                pred_highlight_mask=pred_highlight_mask,
                args=args,
            )

            if lpips_model is not None:
                metrics["lpips_full"] = compute_lpips_distance(lpips_model, pred_rgb, gt_rgb, args.device)

                fg_mask_rgb = foreground_mask[..., None]
                pred_fg = pred_rgb * fg_mask_rgb + (1.0 - fg_mask_rgb)
                gt_fg = gt_rgb * fg_mask_rgb + (1.0 - fg_mask_rgb)
                metrics["lpips_foreground"] = compute_lpips_distance(lpips_model, pred_fg, gt_fg, args.device)

                crop_mask = np.clip(gt_highlight_mask + pred_highlight_mask, 0.0, 1.0)
                bbox = bbox_from_mask(crop_mask, padding=args.highlight_crop_padding)
                if bbox is not None:
                    pred_crop = crop_rgb(pred_rgb, bbox)
                    gt_crop = crop_rgb(gt_rgb, bbox)
                    metrics["lpips_highlight_crop"] = compute_lpips_distance(lpips_model, pred_crop, gt_crop, args.device)
                else:
                    metrics["lpips_highlight_crop"] = None

            if args.compute_ssim:
                metrics["full_ssim"] = compute_ssim_value(
                    pred_rgb,
                    gt_rgb,
                    filter_size=args.ssim_filter_size,
                    filter_sigma=args.ssim_filter_sigma,
                )

                fg_mask_rgb = foreground_mask[..., None]
                pred_fg = pred_rgb * fg_mask_rgb + (1.0 - fg_mask_rgb)
                gt_fg = gt_rgb * fg_mask_rgb + (1.0 - fg_mask_rgb)
                metrics["foreground_ssim"] = compute_ssim_value(
                    pred_fg,
                    gt_fg,
                    filter_size=args.ssim_filter_size,
                    filter_sigma=args.ssim_filter_sigma,
                )

                crop_mask = np.clip(gt_highlight_mask + pred_highlight_mask, 0.0, 1.0)
                bbox = bbox_from_mask(crop_mask, padding=args.highlight_crop_padding)
                if bbox is not None:
                    pred_crop = crop_rgb(pred_rgb, bbox)
                    gt_crop = crop_rgb(gt_rgb, bbox)
                    metrics["highlight_crop_ssim"] = compute_ssim_value(
                        pred_crop,
                        gt_crop,
                        filter_size=args.ssim_filter_size,
                        filter_sigma=args.ssim_filter_sigma,
                    )
                else:
                    metrics["highlight_crop_ssim"] = None

            record = {
                "sample_key": sample.get("sample_key")
                or f"{sample.get('preset', 'na')}_{sample.get('object_id', 'unknown')}_v{int(sample.get('view_idx', 0)):03d}_t{int(sample.get('target_lighting_index', 0) or 0):03d}",
                "preset": sample.get("preset"),
                "object_id": sample.get("object_id"),
                "view_idx": view_idx,
                "target_file": sample.get("target_file"),
                "mask_source": mask_source,
                "paths": {
                    "gt": str(gt_path_resolved),
                    "pred": pred_path,
                },
                "metrics": metrics,
            }
            method_records[method_name].append(record)

    for method_name, records in method_records.items():
        by_preset_records = defaultdict(list)
        for record in records:
            by_preset_records[record.get("preset")].append(record)
        results["methods"][method_name] = {
            "sample_count": len(records),
            "overall": summarize_records(records),
            "by_preset": {preset_name: summarize_records(preset_records) for preset_name, preset_records in sorted(by_preset_records.items())},
            "samples": records,
        }

    default_stem = assets_manifest_path.with_suffix("")
    output_json = resolve_repo_path(args.output_json) if args.output_json else Path(str(default_stem) + "_highlight_metrics.json")
    output_md = resolve_repo_path(args.output_md) if args.output_md else Path(str(default_stem) + "_highlight_metrics.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(results), encoding="utf-8")

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
