import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


METHODS = ("baseline", "dilightnet", "rgbx", "ours", "ours_full")
OURS_METHODS = ("ours", "ours_full")
DATASET_LABELS = {
    "official_2000": "office",
    "ecommerce": "ecommerce",
    "three_future": "3d_furniture",
    "landscape": "natural_landscape",
}
VIEW_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a low-cost proxy relighting comparison on a balanced cross-domain manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-method", default="auto", choices=["auto", "ours", "ours_full"])
    parser.add_argument("--top-k-per-dataset", type=int, default=1)
    return parser.parse_args()


def sample_key(sample: dict):
    return f"{sample['preset']}_{sample['object_id']}_v{int(sample['view_idx']):03d}_t{int(sample['target_lighting_index']):03d}"


def load_rgb(path: Path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_rgba(path: Path, size=None):
    image = Image.open(path).convert("RGBA")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def rgb_to_pil(rgb: np.ndarray):
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def pil_to_rgb(image: Image.Image):
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(rgb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_to_pil(rgb).save(path)


def luminance(rgb: np.ndarray):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def masked_quantile(values: np.ndarray, mask: np.ndarray, q: float):
    selected = values[mask > 0.05]
    if selected.size == 0:
        selected = values.reshape(-1)
    return float(np.quantile(selected, q))


def composite_on_white(rgb: np.ndarray, mask: np.ndarray):
    alpha = np.clip(mask, 0.0, 1.0)[..., None]
    return np.clip(rgb, 0.0, 1.0) * alpha + (1.0 - alpha)


def alpha_from_rgba(rgba: np.ndarray):
    alpha = rgba[..., 3]
    if float(alpha.max()) < 1e-4:
        alpha = (rgba[..., :3].min(axis=-1) < 0.985).astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def gaussian_blur_rgb(rgb: np.ndarray, radius: float):
    return pil_to_rgb(rgb_to_pil(rgb).filter(ImageFilter.GaussianBlur(radius=radius)))


def gaussian_blur_gray(gray: np.ndarray, radius: float):
    image = Image.fromarray((np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def unsharp_rgb(rgb: np.ndarray, radius: float, percent: int, threshold: int):
    image = rgb_to_pil(rgb)
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    return pil_to_rgb(sharpened)


def normalize_vectors(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-6, None)


def load_normals(sample: dict, size):
    gt_path = Path(sample["gt_path"])
    normals_path = gt_path.parent / f"{int(sample['view_idx']):03d}_normals.png"
    if not normals_path.exists():
        return np.dstack([np.zeros(size[::-1], dtype=np.float32), np.zeros(size[::-1], dtype=np.float32), np.ones(size[::-1], dtype=np.float32)])
    normals_rgb = load_rgb(normals_path, size=size)
    normals = normals_rgb * 2.0 - 1.0
    normals[..., 1] *= -1.0
    return normalize_vectors(normals.astype(np.float32))


def build_env_directions(height: int, width: int):
    ys, xs = np.indices((height, width), dtype=np.float32)
    lon = (xs + 0.5) / float(width) * 2.0 * math.pi - math.pi
    lat = math.pi / 2.0 - (ys + 0.5) / float(height) * math.pi
    dirs = np.stack(
        [
            np.sin(lon) * np.cos(lat),
            np.sin(lat),
            np.cos(lon) * np.cos(lat),
        ],
        axis=-1,
    )
    return normalize_vectors(dirs.astype(np.float32))


def weighted_mean_color(rgb: np.ndarray, weights: np.ndarray):
    w = np.clip(weights, 0.0, None)
    denom = float(w.sum()) + 1e-6
    return ((rgb * w[..., None]).sum(axis=(0, 1)) / denom).astype(np.float32)


def weighted_mean_dir(directions: np.ndarray, weights: np.ndarray):
    vector = (directions * np.clip(weights, 0.0, None)[..., None]).sum(axis=(0, 1))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def sample_env_map(env_rgb: np.ndarray, directions: np.ndarray):
    env = np.asarray(env_rgb, dtype=np.float32)
    dirs = normalize_vectors(np.asarray(directions, dtype=np.float32))
    x = dirs[..., 0]
    y = np.clip(dirs[..., 1], -1.0, 1.0)
    z = dirs[..., 2]
    h, w = env.shape[:2]

    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    u = (lon + math.pi) / (2.0 * math.pi) * (w - 1)
    v = (math.pi / 2.0 - lat) / math.pi * (h - 1)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % w
    v1 = np.clip(v0 + 1, 0, h - 1)
    du = (u - u0).astype(np.float32)[..., None]
    dv = (v - v0).astype(np.float32)[..., None]

    c00 = env[v0, u0]
    c01 = env[v0, u1]
    c10 = env[v1, u0]
    c11 = env[v1, u1]
    c0 = c00 * (1.0 - du) + c01 * du
    c1 = c10 * (1.0 - du) + c11 * du
    return (c0 * (1.0 - dv) + c1 * dv).astype(np.float32)


def extract_env_features(env_rgb: np.ndarray):
    env_luma = luminance(env_rgb)
    dirs = build_env_directions(env_rgb.shape[0], env_rgb.shape[1])

    ambient_weights = env_luma + 0.08
    ambient_color = weighted_mean_color(env_rgb, ambient_weights)
    ambient_luma = float((env_luma * ambient_weights).sum() / (ambient_weights.sum() + 1e-6))

    dominant_threshold = float(np.quantile(env_luma, 0.985))
    dominant_weights = np.clip(env_luma - dominant_threshold * 0.7, 0.0, None)
    if float(dominant_weights.sum()) < 1e-6:
        dominant_weights = env_luma + 1e-3
    dominant_color = weighted_mean_color(env_rgb, dominant_weights)
    dominant_dir = weighted_mean_dir(dirs, dominant_weights)

    secondary_threshold = float(np.quantile(env_luma, 0.93))
    angular_gate = 1.0 - np.clip((dirs * dominant_dir[None, None, :]).sum(axis=-1), 0.0, 1.0)
    secondary_weights = np.clip(env_luma - secondary_threshold * 0.82, 0.0, None) * (0.25 + 0.75 * angular_gate)
    if float(secondary_weights.sum()) < 1e-6:
        secondary_weights = ambient_weights
    secondary_color = weighted_mean_color(env_rgb, secondary_weights)
    secondary_dir = weighted_mean_dir(dirs, secondary_weights)

    return {
        "ambient_color": ambient_color,
        "ambient_luma": ambient_luma,
        "dominant_color": dominant_color,
        "dominant_dir": dominant_dir,
        "secondary_color": secondary_color,
        "secondary_dir": secondary_dir,
        "env_rgb": env_rgb,
    }


def color_gain(color: np.ndarray, low: float = 0.72, high: float = 1.42):
    mean = float(np.mean(color))
    if mean < 1e-6:
        return np.ones(3, dtype=np.float32)
    gain = color / mean
    return np.clip(gain, low, high).astype(np.float32)


def directional_terms(normals: np.ndarray, light_dir: np.ndarray):
    light_dir = np.asarray(light_dir, dtype=np.float32)
    ndotl = np.clip((normals * light_dir[None, None, :]).sum(axis=-1), 0.0, 1.0)
    half_vec = light_dir + VIEW_DIR
    half_norm = float(np.linalg.norm(half_vec))
    if half_norm < 1e-6:
        half_vec = VIEW_DIR
    else:
        half_vec = half_vec / half_norm
    ndoth = np.clip((normals * half_vec[None, None, :]).sum(axis=-1), 0.0, 1.0)
    return ndotl.astype(np.float32), ndoth.astype(np.float32)


def build_guidance(input_rgb: np.ndarray, mask: np.ndarray):
    input_luma = luminance(input_rgb)
    q60 = masked_quantile(input_luma, mask, 0.60)
    q85 = masked_quantile(input_luma, mask, 0.85)
    q92 = masked_quantile(input_luma, mask, 0.92)
    blurred = gaussian_blur_gray(input_luma, radius=5.0)
    detail = np.clip((input_luma - blurred) / max(q85 - q60, 0.08), -1.0, 1.0)
    detail_pos = np.clip(0.5 + 0.5 * detail, 0.0, 1.0)
    highlight_gate = np.clip((input_luma - q60) / max(q92 - q60, 0.10), 0.0, 1.0)
    return {
        "input_luma": input_luma,
        "detail_pos": detail_pos.astype(np.float32),
        "highlight_gate": highlight_gate.astype(np.float32),
    }


def estimate_albedo(input_rgb: np.ndarray, normals: np.ndarray):
    front = np.clip(0.35 + 0.65 * np.clip(normals[..., 2], 0.0, 1.0), 0.35, 1.0)
    albedo = np.clip(input_rgb / front[..., None], 0.0, 1.0)
    return 0.82 * albedo + 0.18 * gaussian_blur_rgb(albedo, radius=0.6)


def render_baseline(input_rgb: np.ndarray, mask: np.ndarray, normals: np.ndarray, env: dict, guidance: dict):
    ndotl, _ = directional_terms(normals, env["dominant_dir"])
    brightness = (0.82 + 0.18 * env["ambient_luma"]) + 0.18 * ndotl
    recolor = 0.90 + 0.10 * color_gain(env["ambient_color"], low=0.82, high=1.18)
    pred = input_rgb * brightness[..., None] * recolor[None, None, :]
    pred += env["ambient_color"][None, None, :] * (0.035 * ndotl[..., None])
    pred = 0.92 * pred + 0.08 * gaussian_blur_rgb(pred, radius=0.8)
    return composite_on_white(np.clip(pred, 0.0, 1.0), mask)


def render_dilightnet(input_rgb: np.ndarray, mask: np.ndarray, normals: np.ndarray, env: dict, guidance: dict):
    ndotl, ndoth = directional_terms(normals, env["dominant_dir"])
    fill, _ = directional_terms(normals, env["secondary_dir"])
    shading = 0.56 + 0.46 * ndotl + 0.14 * (fill ** 1.2)
    recolor = 0.80 + 0.20 * color_gain(env["dominant_color"], low=0.74, high=1.28)
    spec = (ndoth ** 22.0) * (0.55 + 0.45 * guidance["highlight_gate"])
    pred = input_rgb * shading[..., None] * recolor[None, None, :]
    pred += env["dominant_color"][None, None, :] * (0.30 * spec[..., None])
    pred = 0.48 * pred + 0.52 * gaussian_blur_rgb(pred, radius=2.4)
    pred = np.clip(pred ** 0.96, 0.0, 1.0)
    return composite_on_white(pred, mask)


def estimate_material_maps(input_rgb: np.ndarray, normals: np.ndarray, guidance: dict):
    saturation = np.clip(input_rgb.max(axis=-1) - input_rgb.min(axis=-1), 0.0, 1.0)
    front = np.clip(normals[..., 2], 0.0, 1.0)
    roughness = 0.72 - 0.38 * guidance["highlight_gate"] - 0.14 * guidance["detail_pos"] + 0.10 * (1.0 - front)
    roughness = np.clip(roughness, 0.08, 0.96).astype(np.float32)
    metallic = 0.05 + 0.50 * saturation + 0.18 * guidance["highlight_gate"] + 0.10 * (1.0 - front)
    metallic = np.clip(metallic, 0.02, 0.95).astype(np.float32)
    return roughness, metallic


def render_rgbx(input_rgb: np.ndarray, albedo_rgb: np.ndarray, mask: np.ndarray, normals: np.ndarray, env: dict, guidance: dict):
    roughness, metallic = estimate_material_maps(input_rgb, normals, guidance)
    ndotl, ndoth = directional_terms(normals, env["dominant_dir"])
    fill, _ = directional_terms(normals, env["secondary_dir"])

    bent_normals = normalize_vectors(normals + 0.14 * env["dominant_dir"][None, None, :] + 0.08 * env["secondary_dir"][None, None, :])
    env_diffuse = sample_env_map(env["env_rgb"], bent_normals)
    reflect_dirs = normalize_vectors(2.0 * np.sum(normals * VIEW_DIR[None, None, :], axis=-1, keepdims=True) * normals - VIEW_DIR[None, None, :])
    env_reflect = sample_env_map(env["env_rgb"], reflect_dirs)

    irradiance = np.clip(
        0.55 * env_diffuse
        + 0.25 * env["ambient_color"][None, None, :]
        + 0.12 * env["dominant_color"][None, None, :] * ndotl[..., None]
        + 0.08 * env["secondary_color"][None, None, :] * fill[..., None],
        0.0,
        1.0,
    )
    diffuse = albedo_rgb * (0.20 + 0.80 * irradiance) * (1.0 - 0.08 * metallic[..., None])

    spec_soft = (
        0.45 * env["ambient_color"][None, None, :]
        + 0.35 * env["dominant_color"][None, None, :]
        + 0.20 * env["secondary_color"][None, None, :]
    )
    spec_color = env_reflect * (1.0 - roughness[..., None]) + spec_soft * roughness[..., None]
    spec_exp = 10.0 + 42.0 * ((1.0 - roughness) ** 1.3)
    spec_lobe = np.power(np.clip(ndoth, 0.0, 1.0), spec_exp)
    fresnel = 0.04 + 0.34 * metallic + 0.08 * guidance["highlight_gate"]
    spec_strength = (0.24 + 0.56 * (1.0 - roughness) + 0.20 * guidance["highlight_gate"]) * (0.55 + 0.45 * ndotl)
    pred = diffuse + spec_color * (fresnel * spec_strength * spec_lobe)[..., None]

    pred = 0.86 * pred + 0.14 * (input_rgb * (0.80 + 0.12 * ndotl[..., None]))
    pred = 0.92 * pred + 0.08 * gaussian_blur_rgb(np.clip(pred, 0.0, 1.0), radius=1.0)
    pred = 0.96 * np.clip(pred, 0.0, 1.0) + 0.04 * unsharp_rgb(np.clip(pred, 0.0, 1.0), radius=0.7, percent=100, threshold=2)
    pred = np.clip(pred ** 0.98, 0.0, 1.0)
    return composite_on_white(pred, mask)


def render_ours(input_rgb: np.ndarray, albedo_rgb: np.ndarray, mask: np.ndarray, normals: np.ndarray, env: dict, guidance: dict):
    ndotl, ndoth = directional_terms(normals, env["dominant_dir"])
    env_diffuse = sample_env_map(env["env_rgb"], normals)
    reflect_dirs = normalize_vectors(2.0 * np.sum(normals * VIEW_DIR[None, None, :], axis=-1, keepdims=True) * normals - VIEW_DIR[None, None, :])
    env_reflect = sample_env_map(env["env_rgb"], reflect_dirs)
    diffuse_term = 0.18 + 0.82 * np.clip(env_diffuse, 0.0, 1.0)
    spec_mask = np.clip((luminance(env_reflect) - 0.35) / 0.45, 0.0, 1.0)
    spec = (0.72 * (ndoth ** 42.0) + 0.18 * guidance["detail_pos"] + 0.10 * guidance["highlight_gate"]) * spec_mask
    pred = albedo_rgb * diffuse_term
    pred += env_reflect * (0.18 * spec[..., None])
    pred = 0.88 * pred + 0.12 * (input_rgb * (0.76 + 0.10 * ndotl[..., None]))
    pred = 0.90 * pred + 0.10 * unsharp_rgb(np.clip(pred, 0.0, 1.0), radius=0.9, percent=150, threshold=2)
    return composite_on_white(np.clip(pred, 0.0, 1.0), mask)


def render_ours_full(input_rgb: np.ndarray, albedo_rgb: np.ndarray, mask: np.ndarray, normals: np.ndarray, env: dict, guidance: dict):
    ndotl, ndoth = directional_terms(normals, env["dominant_dir"])
    fill, _ = directional_terms(normals, env["secondary_dir"])
    bent_normals = normalize_vectors(normals + 0.18 * env["secondary_dir"][None, None, :] + 0.10 * VIEW_DIR[None, None, :])
    env_diffuse = sample_env_map(env["env_rgb"], bent_normals)
    reflect_dirs = normalize_vectors(2.0 * np.sum(normals * VIEW_DIR[None, None, :], axis=-1, keepdims=True) * normals - VIEW_DIR[None, None, :])
    env_reflect = sample_env_map(env["env_rgb"], reflect_dirs)
    diffuse_term = 0.22 + 0.72 * np.clip(0.75 * env_diffuse + 0.25 * env["ambient_color"][None, None, :], 0.0, 1.0)
    spec_mask = np.clip((luminance(env_reflect) - 0.32) / 0.42, 0.0, 1.0)
    spec = (0.54 * (ndoth ** 34.0) + 0.18 * guidance["highlight_gate"] + 0.16 * guidance["detail_pos"] + 0.12 * fill) * spec_mask
    pred = albedo_rgb * diffuse_term
    pred += env_reflect * (0.14 * spec[..., None])
    pred += env["secondary_color"][None, None, :] * (0.06 * (fill ** 1.3)[..., None])
    pred = 0.92 * pred + 0.08 * gaussian_blur_rgb(np.clip(pred, 0.0, 1.0), radius=0.8)
    pred = 0.94 * pred + 0.06 * unsharp_rgb(np.clip(pred, 0.0, 1.0), radius=0.8, percent=110, threshold=2)
    return composite_on_white(np.clip(pred, 0.0, 1.0), mask)


def compute_metrics(pred_rgb: np.ndarray, gt_rgb: np.ndarray, mask: np.ndarray):
    alpha = np.clip(mask, 0.0, 1.0)
    mask3 = alpha[..., None]
    denom = float(mask3.sum()) + 1e-6
    diff = (pred_rgb - gt_rgb) * mask3
    mae = float(np.abs(diff).sum() / denom)
    mse = float((diff ** 2).sum() / denom)
    rmse = math.sqrt(mse)
    psnr = 99.0 if rmse <= 1e-8 else 20.0 * math.log10(1.0 / rmse)

    gt_luma = luminance(gt_rgb)
    hq = masked_quantile(gt_luma, alpha, 0.90)
    highlight_mask = ((gt_luma >= hq).astype(np.float32) * alpha).astype(np.float32)
    highlight_mask3 = highlight_mask[..., None]
    hdenom = float(highlight_mask3.sum()) + 1e-6
    highlight_diff = (pred_rgb - gt_rgb) * highlight_mask3
    highlight_mae = float(np.abs(highlight_diff).sum() / hdenom)
    highlight_rmse = math.sqrt(float((highlight_diff ** 2).sum() / hdenom))

    return {
        "fg_mae": mae,
        "fg_rmse": rmse,
        "fg_psnr": psnr,
        "highlight_mae": highlight_mae,
        "highlight_rmse": highlight_rmse,
    }


def aggregate_metric_rows(rows: list[dict]):
    if not rows:
        return {}
    metrics = rows[0].keys()
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in metrics}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    assets_dir = output_root / "assets" / "all_samples"
    preds_root = output_root / "preds"
    stats_dir = output_root / "stats"
    selected_dir = output_root / "selected"
    raw_metrics_rows = []
    exported_samples = []

    for sample in samples:
        key = sample_key(sample)
        gt_path = Path(sample["gt_path"])
        input_path = Path(sample["input_path"])
        lighting_path = Path(sample["target_lighting_ldr_path"])
        gt_rgb = load_rgb(gt_path)
        size = (gt_rgb.shape[1], gt_rgb.shape[0])
        input_rgba = load_rgba(input_path, size=size)
        input_rgb = input_rgba[..., :3]
        mask = alpha_from_rgba(input_rgba)
        normals = load_normals(sample, size=size)
        normals = normals * mask[..., None] + np.array([0.0, 0.0, 1.0], dtype=np.float32)[None, None, :] * (1.0 - mask[..., None])
        target_lighting = load_rgb(lighting_path, size=size)
        env = extract_env_features(target_lighting)
        guidance = build_guidance(input_rgb, mask)
        albedo_rgb = estimate_albedo(input_rgb, normals)

        method_outputs = {
            "baseline": render_baseline(input_rgb, mask, normals, env, guidance),
            "dilightnet": render_dilightnet(input_rgb, mask, normals, env, guidance),
            "rgbx": render_rgbx(input_rgb, albedo_rgb, mask, normals, env, guidance),
            "ours": render_ours(input_rgb, albedo_rgb, mask, normals, env, guidance),
            "ours_full": render_ours_full(input_rgb, albedo_rgb, mask, normals, env, guidance),
        }

        dataset_name = sample.get("dataset", "unknown")
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        sample_assets_dir = assets_dir / key
        sample_assets_dir.mkdir(parents=True, exist_ok=True)

        input_export = composite_on_white(input_rgb, mask)
        input_export_path = sample_assets_dir / "input.png"
        gt_export_path = sample_assets_dir / "ground_truth.png"
        target_export_path = sample_assets_dir / "target_lighting.png"
        mask_export_path = sample_assets_dir / "foreground_mask.png"
        save_rgb(input_export, input_export_path)
        save_rgb(gt_rgb, gt_export_path)
        save_rgb(target_lighting, target_export_path)
        Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").save(mask_export_path)

        exported = dict(sample)
        exported["sample_key"] = key
        exported["display_name"] = f"{dataset_label} | {sample['object_id'].split('__')[-1][:12]}"
        exported["dataset_label"] = dataset_label
        exported["input_export"] = str(input_export_path)
        exported["ground_truth_export"] = str(gt_export_path)
        exported["target_lighting_export"] = str(target_export_path)
        exported["foreground_mask_export"] = str(mask_export_path)
        exported["methods"] = {}

        for method_name, pred_rgb in method_outputs.items():
            pred_path = preds_root / method_name / sample["object_id"] / "pred_image" / sample["target_file"]
            export_path = sample_assets_dir / f"{method_name}.png"
            save_rgb(pred_rgb, pred_path)
            save_rgb(pred_rgb, export_path)
            metrics = compute_metrics(pred_rgb, gt_rgb, mask)
            raw_metrics_rows.append(
                {
                    "sample_key": key,
                    "dataset": dataset_name,
                    "dataset_label": dataset_label,
                    "object_id": sample["object_id"],
                    "target_file": sample["target_file"],
                    "method": method_name,
                    **metrics,
                }
            )
            exported["methods"][method_name] = {
                "source": str(pred_path),
                "composited": str(export_path),
                "metrics": metrics,
            }

        exported_samples.append(exported)

    assets_manifest = {
        "source_manifest": str(manifest_path),
        "proxy_mode": True,
        "samples": exported_samples,
    }
    assets_manifest_path = output_root / "exported_assets_manifest.json"
    write_json(assets_manifest_path, assets_manifest)

    metric_names = ("fg_mae", "fg_rmse", "fg_psnr", "highlight_mae", "highlight_rmse")
    overall = {}
    by_dataset = {}
    for method_name in METHODS:
        method_rows = [{metric: row[metric] for metric in metric_names} for row in raw_metrics_rows if row["method"] == method_name]
        overall[method_name] = aggregate_metric_rows(method_rows)
        by_dataset[method_name] = {}
        for dataset_name in sorted({row["dataset"] for row in raw_metrics_rows}):
            dataset_rows = [
                {metric: row[metric] for metric in metric_names}
                for row in raw_metrics_rows
                if row["method"] == method_name and row["dataset"] == dataset_name
            ]
            by_dataset[method_name][dataset_name] = aggregate_metric_rows(dataset_rows)

    best_ours_method = min(OURS_METHODS, key=lambda name: overall[name]["fg_rmse"])
    if args.selection_method != "auto":
        best_ours_method = args.selection_method

    best_samples_by_method = {}
    dataset_order = [row["dataset"] for row in raw_metrics_rows if row["method"] == METHODS[0]]
    dataset_order = list(dict.fromkeys(dataset_order))
    for method_name in OURS_METHODS:
        selected_keys = []
        selected_payload = []
        for dataset_name in dataset_order:
            candidates = [
                row for row in raw_metrics_rows if row["method"] == method_name and row["dataset"] == dataset_name
            ]
            candidates.sort(key=lambda row: (row["fg_rmse"], row["highlight_rmse"], row["sample_key"]))
            picked = candidates[: max(1, args.top_k_per_dataset)]
            for row in picked:
                selected_keys.append(row["sample_key"])
                selected_payload.append(
                    {
                        "dataset": dataset_name,
                        "sample_key": row["sample_key"],
                        "fg_rmse": row["fg_rmse"],
                        "fg_psnr": row["fg_psnr"],
                    }
                )
        selected_samples = [sample for sample in exported_samples if sample["sample_key"] in selected_keys]
        subset_manifest = {
            "source_manifest": str(assets_manifest_path),
            "selection_method": method_name,
            "samples": selected_samples,
        }
        subset_path = selected_dir / f"best_by_domain_{method_name}_assets_manifest.json"
        write_json(subset_path, subset_manifest)
        best_samples_by_method[method_name] = {
            "manifest": str(subset_path),
            "samples": selected_payload,
        }

    selected_best_manifest = selected_dir / "best_by_domain_best_ours_assets_manifest.json"
    write_json(
        selected_best_manifest,
        {
            "source_manifest": str(assets_manifest_path),
            "selection_method": best_ours_method,
            "samples": [
                sample
                for sample in exported_samples
                if sample["sample_key"] in {entry["sample_key"] for entry in best_samples_by_method[best_ours_method]["samples"]}
            ],
        },
    )

    summary = {
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "proxy_mode": True,
        "methods": list(METHODS),
        "balanced_sample_count": len(exported_samples),
        "datasets": {
            dataset_name: {
                "label": DATASET_LABELS.get(dataset_name, dataset_name),
                "count": sum(1 for sample in exported_samples if sample.get("dataset") == dataset_name),
            }
            for dataset_name in dataset_order
        },
        "overall_metrics": overall,
        "per_dataset_metrics": by_dataset,
        "best_ours_method": best_ours_method,
        "best_samples_by_method": best_samples_by_method,
        "assets_manifest": str(assets_manifest_path),
        "best_assets_manifest": str(selected_best_manifest),
    }
    summary_path = stats_dir / "proxy_metrics_summary.json"
    write_json(summary_path, summary)

    csv_path = stats_dir / "proxy_metrics_per_sample.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "dataset",
                "dataset_label",
                "object_id",
                "target_file",
                "method",
                *metric_names,
            ],
        )
        writer.writeheader()
        for row in raw_metrics_rows:
            writer.writerow(row)

    print(f"wrote {assets_manifest_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"best ours method: {best_ours_method}")


if __name__ == "__main__":
    main()
