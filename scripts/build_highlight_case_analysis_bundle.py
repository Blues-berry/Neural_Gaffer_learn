import argparse
import csv
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.assess_dataset_quality import compute_highlight_local_mean, normalize_rgb


METHOD_LABELS = {
    "baseline": "Neural Gaffer",
    "dilightnet": "DiLightNet",
    "rgbx": "RGB↔X",
    "ours": "Ours",
    "ours_full": "Ours (Full)",
    "baseline_0316_fallback": "0316 Baseline",
    "jbhdfvfc_ckpt80k": "80K Highlight",
    "cosine0331_03": "Cosine 0331-03",
    "xkmlb19f_like_relative_fallback": "Relative Fallback",
    "hyblite_0331_02_fallback": "Abl. Hybrid Lite",
    "officialval_0403_04": "Ours (OfficialVal)",
}

LOWER_IS_BETTER = {
    "lpips_full",
    "lpips_foreground",
    "lpips_highlight_crop",
    "highlight_rmse",
    "highlight_area_abs_error",
    "highlight_saturated_ratio_abs_error",
    "highlight_p95_luma_abs_error",
    "highlight_centroid_distance",
    "highlight_chroma_l1_on_gt_mask",
    "highlight_mse_ratio",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build highlight zoom figures, diagnostic figures, and metric tables for a selected panel page."
    )
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--aggregate-assets-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="*", default=["baseline", "dilightnet", "rgbx", "ours"])
    parser.add_argument("--diagnostic-methods", nargs="*", default=None)
    parser.add_argument("--focus-methods", nargs="*", default=None)
    parser.add_argument("--compact-paper-methods", action="store_true")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--tile-size", type=int, default=280)
    parser.add_argument("--diagnostic-tile-size", type=int, default=220)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=64)
    parser.add_argument("--label-width", type=int, default=260)
    parser.add_argument("--crop-padding", type=int, default=28)
    parser.add_argument("--min-crop-size", type=int, default=96)
    parser.add_argument("--skip-eval-if-present", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--paper-style", action="store_true", default=True)
    parser.add_argument("--sidebar-show-text", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit_font(text: str, max_width: int, start_size: int, min_size: int = 14, bold: bool = False):
    for size in range(start_size, min_size - 1, -1):
        font = load_font(size, bold=bold)
        bbox = font.getbbox(text)
        if (bbox[2] - bbox[0]) <= max_width - 10:
            return font
    return load_font(min_size, bold=bold)


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4, align="center")


def load_rgb_image(path_value: str):
    image = Image.open(path_value).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]):
    target_h, target_w = target_hw
    if mask.shape == (target_h, target_w):
        return mask
    image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image = image.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def crop_array(arr: np.ndarray, bbox):
    top, bottom, left, right = bbox
    if arr.ndim == 2:
        return arr[top:bottom, left:right]
    return arr[top:bottom, left:right, ...]


def crop_to_image(arr: np.ndarray, bbox, size: int):
    cropped = crop_array(arr, bbox)
    if cropped.ndim == 2:
        image = Image.fromarray((np.clip(cropped, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        image = image.resize((size, size), Image.Resampling.NEAREST)
        return image.convert("RGB")
    image = Image.fromarray((np.clip(cropped, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def array_to_image(arr: np.ndarray, size: int, resample=Image.Resampling.BICUBIC):
    if arr.ndim == 2:
        image = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        image = image.resize((size, size), resample)
        return image.convert("RGB")
    image = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize((size, size), resample)
    return image


def array_to_image_hw(arr: np.ndarray, width: int, height: int, resample=Image.Resampling.BICUBIC):
    if arr.ndim == 2:
        image = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        image = image.resize((width, height), resample)
        return image.convert("RGB")
    image = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize((width, height), resample)
    return image


def method_label(name: str):
    return METHOD_LABELS.get(name, name)


def wrap_label_text(text: str, width: int = 18):
    text = str(text or "").strip()
    if not text:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def square_bbox_from_masks(masks: list[np.ndarray], padding: int, min_crop_size: int, image_hw: tuple[int, int]):
    height, width = image_hw
    ys_list = []
    xs_list = []
    for mask in masks:
        if mask is None:
            continue
        ys, xs = np.nonzero(mask > 0.5)
        if ys.size == 0 or xs.size == 0:
            continue
        ys_list.append(ys)
        xs_list.append(xs)
    if not ys_list:
        side = min(max(int(min_crop_size), min(height, width) // 3), min(height, width))
        cy = height / 2.0
        cx = width / 2.0
    else:
        ys = np.concatenate(ys_list)
        xs = np.concatenate(xs_list)
        top = int(ys.min()) - int(padding)
        bottom = int(ys.max()) + int(padding) + 1
        left = int(xs.min()) - int(padding)
        right = int(xs.max()) + int(padding) + 1
        cy = (top + bottom) / 2.0
        cx = (left + right) / 2.0
        side = max(bottom - top, right - left, int(min_crop_size))

    side = min(side, min(height, width))
    half = side / 2.0
    top = int(round(cy - half))
    left = int(round(cx - half))
    bottom = top + side
    right = left + side

    if top < 0:
        bottom -= top
        top = 0
    if left < 0:
        right -= left
        left = 0
    if bottom > height:
        top -= bottom - height
        bottom = height
    if right > width:
        left -= right - width
        right = width
    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, height)
    right = min(right, width)
    return top, bottom, left, right


def compute_primary_highlight_bbox(
    gt_rgb: np.ndarray,
    gt_highlight_mask: np.ndarray,
    padding: int,
    min_crop_size: int,
    max_crop_fraction: float = 0.38,
):
    height, width = gt_rgb.shape[:2]
    binary = np.asarray(gt_highlight_mask, dtype=np.float32) > 0.5
    if not np.any(binary):
        return square_bbox_from_masks([gt_highlight_mask], padding, min_crop_size, (height, width))

    luminance = (
        0.299 * gt_rgb[..., 0]
        + 0.587 * gt_rgb[..., 1]
        + 0.114 * gt_rgb[..., 2]
    ).astype(np.float32)
    weighted = luminance * binary.astype(np.float32)
    peak_flat = int(np.argmax(weighted))
    peak_y, peak_x = np.unravel_index(peak_flat, weighted.shape)

    labels, num_labels = scipy.ndimage.label(binary.astype(np.uint8))
    chosen = int(labels[peak_y, peak_x]) if num_labels > 0 else 0
    if chosen <= 0:
        ys, xs = np.nonzero(binary)
        peak_y = int(np.round(float(ys.mean())))
        peak_x = int(np.round(float(xs.mean())))
        chosen_mask = binary
    else:
        chosen_mask = labels == chosen

    ys, xs = np.nonzero(chosen_mask)
    if ys.size == 0 or xs.size == 0:
        return square_bbox_from_masks([gt_highlight_mask], padding, min_crop_size, (height, width))

    comp_top = int(ys.min()) - int(padding)
    comp_bottom = int(ys.max()) + int(padding) + 1
    comp_left = int(xs.min()) - int(padding)
    comp_right = int(xs.max()) + int(padding) + 1
    comp_h = comp_bottom - comp_top
    comp_w = comp_right - comp_left

    max_side = max(int(min(height, width) * float(max_crop_fraction)), int(min_crop_size))
    side = max(int(min_crop_size), comp_h, comp_w)
    side = min(side, max_side)

    half = side / 2.0
    top = int(round(float(peak_y) - half))
    left = int(round(float(peak_x) - half))
    bottom = top + side
    right = left + side

    if top < 0:
        bottom -= top
        top = 0
    if left < 0:
        right -= left
        left = 0
    if bottom > height:
        top -= bottom - height
        bottom = height
    if right > width:
        left -= right - width
        right = width
    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, height)
    right = min(right, width)
    return top, bottom, left, right


def colorize_scalar_map(values: np.ndarray, mode: str):
    values = np.asarray(values, dtype=np.float32)
    norm = np.clip(values, 0.0, 1.0)
    if mode == "heat":
        r = np.clip(1.3 * norm, 0.0, 1.0)
        g = np.clip(1.1 * np.sqrt(norm), 0.0, 1.0)
        b = np.clip(0.25 * norm, 0.0, 1.0)
    elif mode == "relative":
        r = np.clip(0.15 + 1.1 * norm, 0.0, 1.0)
        g = np.clip(0.05 + 0.8 * np.power(norm, 0.7), 0.0, 1.0)
        b = np.clip(0.2 + 0.45 * np.power(1.0 - norm, 0.5), 0.0, 1.0)
    else:
        r = g = b = norm
    return np.stack([r, g, b], axis=-1)


def normalize_for_display(values: np.ndarray, mask: np.ndarray | None = None, percentile_low: float = 5.0, percentile_high: float = 95.0):
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        valid = arr[np.asarray(mask) > 0.5]
    else:
        valid = arr.reshape(-1)
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(valid, percentile_low))
    hi = float(np.percentile(valid, percentile_high))
    if hi <= lo + 1e-8:
        hi = lo + 1e-8
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 220, 40), background_dim: float = 0.38):
    base = np.asarray(rgb, dtype=np.float32).copy()
    base *= float(background_dim)
    mask = np.asarray(mask, dtype=np.float32)[..., None]
    color_arr = np.asarray(color, dtype=np.float32) / 255.0
    blended = base * (1.0 - mask) + (0.35 * base + 0.65 * color_arr) * mask
    return np.clip(blended, 0.0, 1.0)


def heatmap_overlay(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.72):
    base = np.asarray(rgb, dtype=np.float32)
    overlay = colorize_scalar_map(heatmap, mode="heat")
    return np.clip((1.0 - alpha) * base + alpha * overlay, 0.0, 1.0)


def draw_bbox_overlay(rgb: np.ndarray, bbox, color=(235, 70, 60), width: int = 4):
    image = array_to_image(rgb, size=rgb.shape[0])
    draw = ImageDraw.Draw(image)
    top, bottom, left, right = bbox
    draw.rounded_rectangle((left, top, right - 1, bottom - 1), radius=14, outline=color, width=width)
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_local_relative_brightness(gt_rgb: np.ndarray, foreground_mask: np.ndarray):
    rgb = normalize_rgb(gt_rgb)
    if rgb is None:
        return np.zeros(foreground_mask.shape, dtype=np.float32)
    luminance = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.float32)
    lum_tensor = torch.from_numpy(luminance).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.asarray(foreground_mask, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    local_mean = compute_highlight_local_mean(
        lum_tensor,
        foreground_mask=mask_tensor,
        local_kernel_size=15,
        eps=1e-4,
    )
    rel = lum_tensor - local_mean
    rel = rel.squeeze(0).squeeze(0).cpu().numpy()
    return normalize_for_display(rel, mask=foreground_mask, percentile_low=10.0, percentile_high=99.0)


def mask_tp_fp_fn_overlay(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, background_dim: float = 0.18):
    base = np.asarray(rgb, dtype=np.float32).copy() * float(background_dim)
    gt = np.asarray(gt_mask, dtype=np.float32) > 0.5
    pred = np.asarray(pred_mask, dtype=np.float32) > 0.5
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    vis = base
    vis[tp] = 0.25 * vis[tp] + 0.75 * np.array([0.15, 0.85, 0.20], dtype=np.float32)
    vis[fp] = 0.25 * vis[fp] + 0.75 * np.array([0.92, 0.20, 0.20], dtype=np.float32)
    vis[fn] = 0.25 * vis[fn] + 0.75 * np.array([0.20, 0.45, 0.95], dtype=np.float32)
    return np.clip(vis, 0.0, 1.0)


def compute_highlight_error_map(pred_rgb: np.ndarray, gt_rgb: np.ndarray, gt_highlight_mask: np.ndarray):
    diff = np.mean(np.abs(np.asarray(pred_rgb, dtype=np.float32) - np.asarray(gt_rgb, dtype=np.float32)), axis=-1)
    mask = np.asarray(gt_highlight_mask, dtype=np.float32) > 0.5
    masked = diff * mask.astype(np.float32)
    return masked


def compute_error_scale(panel_samples: list[dict], methods: list[str]):
    values = []
    for sample in panel_samples:
        gt_rgb, _, gt_highlight_mask = resolve_case_images(sample)
        gt_binary = np.asarray(gt_highlight_mask, dtype=np.float32) > 0.5
        if not np.any(gt_binary):
            continue
        for method in methods:
            pred_rgb = get_method_rgb(sample, method)
            err = compute_highlight_error_map(pred_rgb, gt_rgb, gt_highlight_mask)
            vals = err[gt_binary]
            if vals.size:
                values.extend(vals.tolist())
    if not values:
        return 0.25
    return max(0.05, float(np.percentile(np.asarray(values, dtype=np.float32), 95.0)))


def render_error_map_overlay(gt_rgb: np.ndarray, error_map: np.ndarray, gt_highlight_mask: np.ndarray, scale_max: float):
    mask = np.asarray(gt_highlight_mask, dtype=np.float32) > 0.5
    norm = np.clip(np.asarray(error_map, dtype=np.float32) / max(scale_max, 1e-6), 0.0, 1.0)
    base = np.asarray(gt_rgb, dtype=np.float32).copy() * 0.12
    heat = colorize_scalar_map(norm, mode="heat")
    vis = base
    vis[mask] = heat[mask]
    return np.clip(vis, 0.0, 1.0)


def draw_error_colorbar(draw: ImageDraw.ImageDraw, canvas: Image.Image, box, scale_max: float):
    left, top, right, bottom = box
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    bar = colorize_scalar_map(gradient, mode="heat")
    bar_img = array_to_image_hw(bar, width, height)
    canvas.paste(bar_img, (left, top))
    draw.rounded_rectangle((left, top, right, bottom), radius=10, outline=(160, 168, 180), width=2)
    font = load_font(16, bold=False)
    draw.text((left, bottom + 8), "0.00", font=font, fill=(60, 64, 72))
    draw.text((right - 52, bottom + 8), f"{scale_max:.2f}", font=font, fill=(60, 64, 72))
    title_font = load_font(16, bold=True)
    centered_text(draw, (left, top - 28, right, top - 2), "Abs error on GT M_h", title_font, (30, 34, 40))


def resolve_case_images(sample: dict):
    gt_rgb = load_rgb_image(sample["ground_truth_composited_export"])
    foreground_mask = load_mask(sample.get("foreground_mask_export"))
    if foreground_mask is None:
        foreground_mask = np.ones(gt_rgb.shape[:2], dtype=np.float32)
    if foreground_mask.shape != gt_rgb.shape[:2]:
        foreground_mask = resize_mask(foreground_mask, gt_rgb.shape[:2])
    gt_highlight_mask = load_mask(sample.get("gt_highlight_mask_binary_export"))
    if gt_highlight_mask is None:
        gt_highlight_mask = np.zeros(gt_rgb.shape[:2], dtype=np.float32)
    if gt_highlight_mask.shape != gt_rgb.shape[:2]:
        gt_highlight_mask = resize_mask(gt_highlight_mask, gt_rgb.shape[:2])
    return gt_rgb, foreground_mask, gt_highlight_mask


def build_selected_assets_manifest(panel_manifest: dict, output_path: Path):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_panel_manifest": panel_manifest.get("source_manifest"),
        "selection_name": panel_manifest.get("selection_name"),
        "visual_mode": panel_manifest.get("visual_mode"),
        "samples": panel_manifest.get("samples", []),
    }
    dump_json(output_path, payload)
    return payload


def run_eval(assets_manifest: Path, output_root: Path, methods: list[str], device: str, skip_if_present: bool):
    output_json = output_root / "highlight_metrics.json"
    output_md = output_root / "highlight_metrics.md"
    output_csv = output_root / "highlight_metrics_per_sample.csv"
    if skip_if_present and output_json.exists() and output_csv.exists():
        return output_json, output_md, output_csv
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_highlight_metrics_on_assets_manifest.py"),
        "--assets-manifest",
        str(assets_manifest),
        "--methods",
        *methods,
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--output-per-sample-csv",
        str(output_csv),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return output_json, output_md, output_csv


def build_summary_rows(eval_payload: dict, metrics: list[str], methods: list[str]):
    rows = []
    for method in methods:
        method_payload = eval_payload.get("methods", {}).get(method, {})
        overall = method_payload.get("overall", {})
        row = {"method": method, "label": method_label(method)}
        for metric in metrics:
            row[metric] = overall.get(metric, {}).get("mean")
        rows.append(row)
    if metrics:
        primary = metrics[0]
        reverse = primary not in LOWER_IS_BETTER
        rows.sort(
            key=lambda row: (
                float("inf") if row.get(primary) is None and not reverse else float("-inf") if row.get(primary) is None else row.get(primary)
            ),
            reverse=reverse,
        )
    return rows


def format_metric(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-"


def write_table_bundle(output_stem: Path, title: str, metrics: list[str], rows: list[dict]):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [f"# {title}", ""]
    md_lines.append("| method | " + " | ".join(metrics) + " |")
    md_lines.append("| --- | " + " | ".join(["---:"] * len(metrics)) + " |")
    for row in rows:
        md_lines.append("| " + row["label"] + " | " + " | ".join(format_metric(row.get(metric)) for metric in metrics) + " |")
    md_lines.append("")
    (output_stem.with_suffix(".md")).write_text("\n".join(md_lines), encoding="utf-8")

    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "label", *metrics])
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": row["method"], "label": row["label"], **{metric: format_metric(row.get(metric)) for metric in metrics}})


def build_case_table(eval_payload: dict, panel_samples: list[dict], methods: list[str], output_stem: Path):
    sample_meta = {sample["sample_key"]: sample for sample in panel_samples}
    rows = []
    for method in methods:
        for record in eval_payload.get("methods", {}).get(method, {}).get("samples", []):
            sample_key = record.get("sample_key")
            if sample_key not in sample_meta:
                continue
            meta = sample_meta[sample_key]
            metrics = record.get("metrics", {})
            rows.append(
                {
                    "case": f"{meta.get('plain_object_id')} | {meta.get('env_name')}",
                    "method": method_label(method),
                    "full_psnr": metrics.get("full_psnr"),
                    "full_ssim": metrics.get("full_ssim"),
                    "lpips_full": metrics.get("lpips_full"),
                    "highlight_psnr": metrics.get("highlight_psnr"),
                    "highlight_rmse": metrics.get("highlight_rmse"),
                    "highlight_mask_iou": metrics.get("highlight_mask_iou"),
                    "highlight_centroid_distance": metrics.get("highlight_centroid_distance"),
                    "lpips_highlight_crop": metrics.get("lpips_highlight_crop"),
                }
            )

    fields = [
        "case",
        "method",
        "full_psnr",
        "full_ssim",
        "lpips_full",
        "highlight_psnr",
        "highlight_rmse",
        "highlight_mask_iou",
        "highlight_centroid_distance",
        "lpips_highlight_crop",
    ]
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_metric(row[field]) if field not in {"case", "method"} else row[field] for field in fields})

    md_lines = ["# Selected Case Metrics", ""]
    md_lines.append("| " + " | ".join(fields) + " |")
    md_lines.append("| " + " | ".join(["---", "---"] + ["---:"] * (len(fields) - 2)) + " |")
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join([row["case"], row["method"]] + [format_metric(row[field]) for field in fields[2:]])
            + " |"
        )
    md_lines.append("")
    output_stem.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")


def prepare_zoom_columns(methods: list[str]):
    return [("method", method, method_label(method)) for method in methods] + [("gt", "ground_truth", "Ground-truth")]


def prepare_diagnostic_columns(methods: list[str]):
    cols = [
        ("relative", "gt_relative", "GT Local Relative Brightness"),
        ("mask", "gt_highlight", "GT M_h"),
    ]
    for method in methods:
        cols.append(("method_mask", method, f"{method_label(method)} M_h"))
    for method in methods:
        cols.append(("mask_compare", method, f"{method_label(method)} vs GT"))
    for method in methods:
        cols.append(("highlight_absdiff", method, f"|{method_label(method)} - GT| on GT M_h"))
    return cols


def prepare_mask_columns(methods: list[str]):
    cols = [
        ("mask", "gt_highlight", "GT Highlight Mask"),
    ]
    for method in methods:
        cols.append(("method_mask", method, f"{method_label(method)} Highlight Mask"))
    return cols


def prepare_mask_compare_columns(methods: list[str]):
    cols = [
        ("mask", "gt_highlight", "GT Highlight Mask"),
    ]
    for method in methods:
        cols.append(("mask_compare", method, f"{method_label(method)} Mask vs GT"))
    return cols


def prepare_error_columns(methods: list[str]):
    cols = [
        ("mask", "gt_highlight", "GT Highlight Mask"),
    ]
    for method in methods:
        cols.append(("highlight_absdiff", method, f"{method_label(method)} Abs. Error on GT Highlight Region"))
    return cols


def get_method_rgb(sample: dict, method_name: str):
    return load_rgb_image(sample["methods"][method_name]["composited"])


def get_method_mask(sample: dict, method_name: str, target_hw: tuple[int, int]):
    method_entry = sample.get("methods", {}).get(method_name, {})
    mask = load_mask(method_entry.get("highlight_mask_binary"))
    if mask is None:
        return np.zeros(target_hw, dtype=np.float32)
    if mask.shape != target_hw:
        mask = resize_mask(mask, target_hw)
    return mask


def get_input_rgb(sample: dict):
    path = sample.get("input_white_export") or sample.get("input_export") or sample.get("input_composited_export")
    return load_rgb_image(path)


def render_sample_sidebar(draw, canvas, sample: dict, input_rgb: np.ndarray, bbox, box, label_font, small_font, show_text: bool):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=(248, 249, 252), outline=(229, 233, 240), width=2)

    inner_pad = 14
    available_w = right - left - inner_pad * 2
    thumb_h = min(max(90, (bottom - top - inner_pad * 2) if not show_text else (bottom - top) // 2), 132 if show_text else bottom - top - inner_pad * 2)
    thumb_w = min(available_w, thumb_h)
    thumb_left = left + (right - left - thumb_w) // 2
    thumb_top = top + (bottom - top - thumb_h) // 2 if not show_text else top + inner_pad

    thumb_rgb = draw_bbox_overlay(input_rgb, bbox)
    thumb_img = array_to_image_hw(thumb_rgb, thumb_w, thumb_h)
    canvas.paste(thumb_img, (thumb_left, thumb_top))
    draw.rounded_rectangle((thumb_left, thumb_top, thumb_left + thumb_w, thumb_top + thumb_h), radius=12, outline=(235, 110, 100), width=2)

    if not show_text:
        return

    text_top = thumb_top + thumb_h + 12
    object_name = wrap_label_text(str(sample.get("plain_object_id", sample.get("object_id", "unknown"))).replace("_", " "), width=18)
    env_name = wrap_label_text(str(sample.get("env_name", "lighting")).replace("_", " "), width=18)
    centered_text(draw, (left + 8, text_top, right - 8, bottom - 28), object_name + "\n" + env_name, label_font, (34, 36, 42))
    draw.text((left + 14, bottom - 24), sample.get("target_file", ""), font=small_font, fill=(110, 116, 130))


def render_tile_image(sample: dict, col_type: str, col_key: str, bbox, gt_rgb: np.ndarray, gt_h_crop: np.ndarray, gt_crop: np.ndarray, method_masks: dict[str, np.ndarray], error_scale: float | None):
    if col_type == "mask" and col_key == "gt_highlight":
        vis = mask_overlay(gt_crop, gt_h_crop, color=(255, 216, 70), background_dim=0.22)
    elif col_type == "method_mask":
        method_rgb = crop_array(get_method_rgb(sample, col_key), bbox)
        vis = mask_overlay(method_rgb, crop_array(method_masks[col_key], bbox), color=(90, 205, 255), background_dim=0.22)
    elif col_type == "mask_compare":
        method_rgb = crop_array(get_method_rgb(sample, col_key), bbox)
        pred_h_crop = crop_array(method_masks[col_key], bbox)
        vis = mask_tp_fp_fn_overlay(method_rgb, gt_h_crop, pred_h_crop)
    elif col_type == "highlight_absdiff":
        method_rgb = crop_array(get_method_rgb(sample, col_key), bbox)
        err = compute_highlight_error_map(method_rgb, gt_crop, gt_h_crop)
        vis = render_error_map_overlay(gt_crop, err, gt_h_crop, float(error_scale or 0.25))
    elif col_type == "relative":
        fg_mask = np.ones(gt_rgb.shape[:2], dtype=np.float32)
        local_rel = crop_array(compute_local_relative_brightness(gt_rgb, fg_mask), bbox)
        vis = colorize_scalar_map(local_rel, mode="relative")
    else:
        vis = gt_crop
    return vis


def draw_mask_compare_legend(draw: ImageDraw.ImageDraw, box):
    left, top, right, bottom = box
    font = load_font(16, bold=False)
    label_font = load_font(16, bold=True)
    draw.text((left, top), "Mask vs GT:", font=label_font, fill=(40, 44, 52))
    items = [
        ((0.15, 0.85, 0.20), "Green: match GT highlight region (TP)"),
        ((0.92, 0.20, 0.20), "Red: false positive region (FP)"),
        ((0.20, 0.45, 0.95), "Blue: false negative region (FN)"),
    ]
    y = top + 28
    for rgb, text in items:
        chip_left = left
        chip_top = y + 2
        chip_right = chip_left + 18
        chip_bottom = chip_top + 18
        draw.rounded_rectangle(
            (chip_left, chip_top, chip_right, chip_bottom),
            radius=5,
            fill=tuple(int(v * 255) for v in rgb),
            outline=(120, 128, 140),
            width=1,
        )
        draw.text((chip_right + 10, y), text, font=font, fill=(52, 58, 66))
        y += 28


def draw_mask_legend(draw: ImageDraw.ImageDraw, box):
    left, top, right, bottom = box
    font = load_font(16, bold=False)
    label_font = load_font(16, bold=True)
    draw.text((left, top), "Highlight mask:", font=label_font, fill=(40, 44, 52))
    items = [
        ((255, 216, 70), "GT highlight mask"),
        ((90, 205, 255), "Predicted highlight mask"),
    ]
    y = top + 28
    for rgb, text in items:
        chip_left = left
        chip_top = y + 2
        chip_right = chip_left + 18
        chip_bottom = chip_top + 18
        draw.rounded_rectangle(
            (chip_left, chip_top, chip_right, chip_bottom),
            radius=5,
            fill=rgb,
            outline=(120, 128, 140),
            width=1,
        )
        draw.text((chip_right + 10, y), text, font=font, fill=(52, 58, 66))
        y += 28


def draw_error_legend(draw: ImageDraw.ImageDraw, canvas: Image.Image, box, scale_max: float):
    left, top, right, bottom = box
    colorbar_bottom = min(top + 28, bottom)
    draw_error_colorbar(draw, canvas, (left, top, right, colorbar_bottom), scale_max)


def draw_structured_panel(
    panel_samples: list[dict],
    columns: list[tuple[str, str, str]],
    output_path: Path,
    args,
    legend_mode: str | None = None,
):
    tile = int(args.diagnostic_tile_size)
    padding = int(args.padding)
    label_w = int(args.label_width)
    header_h = int(args.header_height)
    left = 24
    top = 18
    rows = len(panel_samples)
    cols = len(columns)
    legend_h = 0 if legend_mode is None else 96
    total_main_cols = cols + 1  # ROI + requested diagnostic columns
    width = left * 2 + label_w + total_main_cols * tile + (total_main_cols - 1) * padding
    height = top * 2 + header_h + rows * tile + (rows - 1) * padding + legend_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    label_font = load_font(18, bold=False)
    small_font = load_font(15, bold=False)
    error_methods = [col_key for col_type, col_key, _ in columns if col_type == "highlight_absdiff"]
    error_scale = compute_error_scale(panel_samples, error_methods) if error_methods else None

    roi_font = fit_font("ROI", tile, 22, min_size=14, bold=True)
    centered_text(draw, (left + label_w, top, left + label_w + tile, top + header_h - 8), "ROI", roi_font, (20, 20, 24))

    x = left + label_w + tile + padding
    for _, _, label in columns:
        font = fit_font(label, tile, 22, min_size=12, bold=True)
        centered_text(draw, (x, top, x + tile, top + header_h - 8), label, font, (20, 20, 24))
        x += tile + padding

    y = top + header_h
    for sample in panel_samples:
        gt_rgb, foreground_mask, gt_highlight_mask = resolve_case_images(sample)
        input_rgb = get_input_rgb(sample)
        method_names = [col_key for col_type, col_key, _ in columns if col_type in {"method_mask", "mask_compare", "highlight_absdiff"}]
        method_masks = {method: get_method_mask(sample, method, gt_rgb.shape[:2]) for method in set(method_names)}
        bbox = compute_primary_highlight_bbox(
            gt_rgb=gt_rgb,
            gt_highlight_mask=gt_highlight_mask,
            padding=args.crop_padding,
            min_crop_size=args.min_crop_size,
        )

        label_box = (left, y, left + label_w - padding, y + tile)
        render_sample_sidebar(draw, canvas, sample, input_rgb, bbox, label_box, label_font, small_font, show_text=args.sidebar_show_text)

        gt_crop = crop_array(gt_rgb, bbox)
        gt_h_crop = crop_array(gt_highlight_mask, bbox)

        roi_img = crop_to_image(input_rgb, bbox, tile)
        canvas.paste(roi_img, (left + label_w, y))
        draw.rounded_rectangle((left + label_w, y, left + label_w + tile, y + tile), radius=12, outline=(225, 228, 235), width=2)

        x = left + label_w + tile + padding
        for col_type, col_key, _ in columns:
            vis = render_tile_image(sample, col_type, col_key, bbox, gt_rgb, gt_h_crop, gt_crop, method_masks, error_scale)
            tile_img = crop_to_image(vis, (0, vis.shape[0], 0, vis.shape[1]), tile)
            canvas.paste(tile_img, (x, y))
            draw.rounded_rectangle((x, y, x + tile, y + tile), radius=12, outline=(225, 228, 235), width=2)
            x += tile + padding
        y += tile + padding

    if legend_mode is not None:
        legend_left = left + label_w + tile + max((cols * tile + (cols - 1) * padding - 520) // 2, 0)
        legend_top = height - legend_h + 14
        if legend_mode == "mask_compare":
            draw_mask_compare_legend(draw, (legend_left, legend_top, legend_left + 520, legend_top + legend_h - 12))
        elif legend_mode == "mask":
            draw_mask_legend(draw, (legend_left, legend_top, legend_left + 520, legend_top + legend_h - 12))
        elif legend_mode == "error":
            draw_error_legend(draw, canvas, (legend_left, legend_top, legend_left + 520, legend_top + legend_h - 12), float(error_scale or 0.25))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def draw_zoom_panel(panel_samples: list[dict], methods: list[str], output_path: Path, args):
    columns = prepare_zoom_columns(methods)
    tile = int(args.tile_size)
    padding = int(args.padding)
    label_w = int(args.label_width)
    header_h = int(args.header_height)
    left = 24
    top = 18
    rows = len(panel_samples)
    cols = len(columns)
    width = left * 2 + label_w + cols * tile + (cols - 1) * padding
    height = top * 2 + header_h + rows * tile + (rows - 1) * padding
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = load_font(20, bold=True)
    label_font = load_font(18, bold=False)
    small_font = load_font(15, bold=False)

    x = left + label_w
    for _, _, label in columns:
        font = fit_font(label, tile, 24, min_size=14, bold=True)
        centered_text(draw, (x, top, x + tile, top + header_h - 8), label, font, (20, 20, 24))
        x += tile + padding

    roi_summary = []
    y = top + header_h
    for sample in panel_samples:
        gt_rgb, foreground_mask, gt_highlight_mask = resolve_case_images(sample)
        input_rgb = get_input_rgb(sample)
        bbox = compute_primary_highlight_bbox(
            gt_rgb=gt_rgb,
            gt_highlight_mask=gt_highlight_mask,
            padding=args.crop_padding,
            min_crop_size=args.min_crop_size,
        )
        roi_summary.append({"sample_key": sample["sample_key"], "bbox_tblr": list(map(int, bbox))})

        label_box = (left, y, left + label_w - padding, y + tile)
        render_sample_sidebar(draw, canvas, sample, input_rgb, bbox, label_box, label_font, small_font, show_text=args.sidebar_show_text)

        x = left + label_w
        for col_type, col_key, _ in columns:
            if col_type == "method":
                rgb = get_method_rgb(sample, col_key)
                tile_img = crop_to_image(rgb, bbox, tile)
            else:
                rgb = gt_rgb
                tile_img = crop_to_image(rgb, bbox, tile)
            canvas.paste(tile_img, (x, y))
            draw.rounded_rectangle((x, y, x + tile, y + tile), radius=12, outline=(225, 228, 235), width=2)
            x += tile + padding
        y += tile + padding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return roi_summary


def draw_diagnostic_panel(panel_samples: list[dict], methods: list[str], output_path: Path, args):
    columns = prepare_diagnostic_columns(methods)
    tile = int(args.diagnostic_tile_size)
    padding = int(args.padding)
    label_w = int(args.label_width)
    header_h = int(args.header_height)
    left = 24
    top = 18
    rows = len(panel_samples)
    cols = len(columns)
    legend_h = 88
    width = left * 2 + label_w + cols * tile + (cols - 1) * padding
    height = top * 2 + header_h + rows * tile + (rows - 1) * padding + legend_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    label_font = load_font(18, bold=False)
    small_font = load_font(15, bold=False)
    error_scale = compute_error_scale(panel_samples, methods)

    x = left + label_w
    for _, _, label in columns:
        font = fit_font(label, tile, 22, min_size=14, bold=True)
        centered_text(draw, (x, top, x + tile, top + header_h - 8), label, font, (20, 20, 24))
        x += tile + padding

    y = top + header_h
    for sample in panel_samples:
        gt_rgb, foreground_mask, gt_highlight_mask = resolve_case_images(sample)
        input_rgb = get_input_rgb(sample)
        method_masks = {method: get_method_mask(sample, method, gt_rgb.shape[:2]) for method in methods}
        bbox = compute_primary_highlight_bbox(
            gt_rgb=gt_rgb,
            gt_highlight_mask=gt_highlight_mask,
            padding=args.crop_padding,
            min_crop_size=args.min_crop_size,
        )

        label_box = (left, y, left + label_w - padding, y + tile)
        render_sample_sidebar(draw, canvas, sample, input_rgb, bbox, label_box, label_font, small_font, show_text=args.sidebar_show_text)

        gt_crop = crop_array(gt_rgb, bbox)
        fg_crop = crop_array(foreground_mask, bbox)
        gt_h_crop = crop_array(gt_highlight_mask, bbox)
        local_rel = crop_array(compute_local_relative_brightness(gt_rgb, foreground_mask), bbox)
        local_rel_rgb = colorize_scalar_map(local_rel, mode="relative")
        local_rel_rgb = crop_to_image(local_rel_rgb, (0, local_rel_rgb.shape[0], 0, local_rel_rgb.shape[1]), tile)

        x = left + label_w
        for col_type, col_key, _ in columns:
            if col_type == "relative":
                vis = np.asarray(local_rel_rgb, dtype=np.float32) / 255.0
            elif col_type == "mask" and col_key == "gt_highlight":
                vis = mask_overlay(gt_crop, gt_h_crop, color=(255, 216, 70), background_dim=0.22)
            elif col_type == "method_mask":
                method_rgb = crop_array(get_method_rgb(sample, col_key), bbox)
                vis = mask_overlay(method_rgb, crop_array(method_masks[col_key], bbox), color=(90, 205, 255), background_dim=0.22)
            elif col_type == "mask_compare":
                method_rgb = crop_array(get_method_rgb(sample, col_key), bbox)
                pred_h_crop = crop_array(method_masks[col_key], bbox)
                vis = mask_tp_fp_fn_overlay(method_rgb, gt_h_crop, pred_h_crop)
            elif col_type == "highlight_absdiff":
                method_rgb = crop_array(get_method_rgb(sample, col_key), bbox)
                err = compute_highlight_error_map(method_rgb, gt_crop, gt_h_crop)
                vis = render_error_map_overlay(gt_crop, err, gt_h_crop, error_scale)
            else:
                vis = gt_crop
            if isinstance(vis, Image.Image):
                tile_img = vis.resize((tile, tile), Image.Resampling.BICUBIC).convert("RGB")
            else:
                tile_img = crop_to_image(vis, (0, vis.shape[0], 0, vis.shape[1]), tile)
            canvas.paste(tile_img, (x, y))
            draw.rounded_rectangle((x, y, x + tile, y + tile), radius=12, outline=(225, 228, 235), width=2)
            x += tile + padding
        y += tile + padding

    legend_left = left + label_w + max((cols * tile + (cols - 1) * padding - 420) // 2, 0)
    legend_top = height - legend_h + 18
    draw_error_colorbar(draw, canvas, (legend_left, legend_top, legend_left + 420, legend_top + 26), error_scale)
    legend_font = load_font(15, bold=False)
    draw.text((legend_left, legend_top + 42), "Mask compare: green = TP, red = FP, blue = FN", font=legend_font, fill=(52, 58, 66))
    draw.text((legend_left + 230, legend_top + 42), "All error maps use the same scale within this figure", font=legend_font, fill=(52, 58, 66))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def draw_split_diagnostic_panels(panel_samples: list[dict], methods: list[str], figures_dir: Path, args):
    draw_structured_panel(
        panel_samples,
        prepare_mask_columns(methods),
        figures_dir / "highlight_mask_headers.png",
        args,
        legend_mode="mask",
    )
    draw_structured_panel(
        panel_samples,
        prepare_mask_compare_columns(methods),
        figures_dir / "highlight_mask_vs_gt_headers.png",
        args,
        legend_mode="mask_compare",
    )
    draw_structured_panel(
        panel_samples,
        prepare_error_columns(methods),
        figures_dir / "highlight_error_on_gt_headers.png",
        args,
        legend_mode="error",
    )


def main():
    args = parse_args()
    panel_manifest_path = Path(args.panel_manifest)
    aggregate_assets_manifest = Path(args.aggregate_assets_manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    panel_manifest = load_json(panel_manifest_path)
    panel_samples = list(panel_manifest.get("samples", []))
    if not panel_samples:
        raise RuntimeError(f"No samples found in {panel_manifest_path}")

    methods = [method for method in args.methods if any(method in sample.get("methods", {}) for sample in panel_samples)]
    if args.focus_methods:
        methods = [method for method in args.focus_methods if method in methods]
    elif args.compact_paper_methods and args.paper_style and "baseline" in methods and "ours" in methods:
        methods = ["baseline", "ours"]

    if args.diagnostic_methods is None:
        diagnostic_methods = list(methods)
    else:
        diagnostic_methods = [method for method in args.diagnostic_methods if method in methods]

    copied_panel_manifest = output_root / "selected_panel_manifest.json"
    copied_panel_manifest.write_text(panel_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    selected_assets_manifest_path = output_root / "selected_assets_manifest.json"
    build_selected_assets_manifest(panel_manifest, selected_assets_manifest_path)

    tables_dir = output_root / "tables"
    if not args.figures_only:
        evaluation_dir = output_root / "evaluation"
        full_eval_dir = evaluation_dir / "full_experiment"
        selected_eval_dir = evaluation_dir / "selected_page"
        full_eval_json, _, _ = run_eval(
            aggregate_assets_manifest,
            full_eval_dir,
            methods,
            args.eval_device,
            args.skip_eval_if_present,
        )
        selected_eval_json, _, _ = run_eval(
            selected_assets_manifest_path,
            selected_eval_dir,
            methods,
            args.eval_device,
            args.skip_eval_if_present,
        )

        full_eval = load_json(full_eval_json)
        selected_eval = load_json(selected_eval_json)

        global_metrics = ["full_psnr", "full_ssim", "lpips_full"]
        foreground_metrics = ["foreground_psnr", "foreground_ssim", "lpips_foreground"]
        highlight_metrics = [
            "highlight_psnr",
            "highlight_rmse",
            "highlight_mask_iou",
            "highlight_area_abs_error",
            "highlight_saturated_ratio_abs_error",
            "highlight_p95_luma_abs_error",
        ]
        diagnostic_metrics = [
            "highlight_centroid_distance",
            "highlight_chroma_l1_on_gt_mask",
            "highlight_mse_ratio",
            "highlight_crop_ssim",
            "lpips_highlight_crop",
        ]

        write_table_bundle(
            tables_dir / "full_experiment_global_quality_table",
            "Full Experiment Global Quality Table",
            global_metrics,
            build_summary_rows(full_eval, global_metrics, methods),
        )
        write_table_bundle(
            tables_dir / "full_experiment_foreground_quality_table",
            "Full Experiment Foreground Quality Table",
            foreground_metrics,
            build_summary_rows(full_eval, foreground_metrics, methods),
        )
        write_table_bundle(
            tables_dir / "full_experiment_highlight_quality_table",
            "Full Experiment Highlight Quality Table",
            highlight_metrics,
            build_summary_rows(full_eval, highlight_metrics, methods),
        )
        write_table_bundle(
            tables_dir / "full_experiment_diagnostic_quality_table",
            "Full Experiment Diagnostic Quality Table",
            diagnostic_metrics,
            build_summary_rows(full_eval, diagnostic_metrics, methods),
        )

        write_table_bundle(
            tables_dir / "selected_page_global_quality_table",
            "Selected Page Global Quality Table",
            global_metrics,
            build_summary_rows(selected_eval, global_metrics, methods),
        )
        write_table_bundle(
            tables_dir / "selected_page_highlight_quality_table",
            "Selected Page Highlight Quality Table",
            highlight_metrics,
            build_summary_rows(selected_eval, highlight_metrics, methods),
        )
        write_table_bundle(
            tables_dir / "selected_page_diagnostic_quality_table",
            "Selected Page Diagnostic Quality Table",
            diagnostic_metrics,
            build_summary_rows(selected_eval, diagnostic_metrics, methods),
        )
        build_case_table(selected_eval, panel_samples, methods, tables_dir / "selected_page_case_metrics")

    figures_dir = output_root / "figures"
    roi_summary = draw_zoom_panel(panel_samples, methods, figures_dir / "highlight_zoom_headers.png", args)
    draw_split_diagnostic_panels(panel_samples, diagnostic_methods, figures_dir, args)
    dump_json(output_root / "roi_summary.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "rois": roi_summary})

    readme_lines = [
        "# Highlight Case Analysis Bundle",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- panel_manifest: {panel_manifest_path}",
        f"- aggregate_assets_manifest: {aggregate_assets_manifest}",
        f"- sample_count: {len(panel_samples)}",
        f"- methods: {', '.join(method_label(m) for m in methods)}",
        "",
        "## Figures",
        "",
        f"- highlight_zoom_headers.png: {figures_dir / 'highlight_zoom_headers.png'}",
        f"- highlight_mask_headers.png: {figures_dir / 'highlight_mask_headers.png'}",
        f"- highlight_mask_vs_gt_headers.png: {figures_dir / 'highlight_mask_vs_gt_headers.png'}",
        f"- highlight_error_on_gt_headers.png: {figures_dir / 'highlight_error_on_gt_headers.png'}",
        "",
    ]
    if args.figures_only:
        readme_lines.extend(
            [
                "## Tables",
                "",
                "- skipped in this run (`--figures-only`)",
                "",
            ]
        )
    else:
        readme_lines.extend(
            [
                "## Tables",
                "",
                f"- {tables_dir / 'full_experiment_global_quality_table.md'}",
                f"- {tables_dir / 'full_experiment_highlight_quality_table.md'}",
                f"- {tables_dir / 'selected_page_global_quality_table.md'}",
                f"- {tables_dir / 'selected_page_highlight_quality_table.md'}",
                f"- {tables_dir / 'selected_page_case_metrics.md'}",
                "",
            ]
        )
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
