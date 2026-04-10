import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.foreground_mask_utils import fallback_white_background_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize scene-background composited assets from a white-background assets manifest."
    )
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--foreground-background-threshold", type=float, default=0.96)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_rgb(path: str | Path):
    return Image.open(path).convert("RGB")


def resize_mask(mask: np.ndarray, size):
    width, height = size
    mask_img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    if mask_img.size != (width, height):
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def infer_mask(reference_image: Image.Image, background_threshold: float):
    rgb = np.asarray(reference_image.convert("RGB"), dtype=np.float32) / 255.0
    mask = fallback_white_background_mask(rgb, background_threshold=background_threshold)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.clip(mask, 0.0, 1.0)


def estimate_alpha_from_white_bg(image: Image.Image, support_mask: np.ndarray):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha = 1.0 - np.min(rgb, axis=-1)
    alpha = np.clip(alpha, 0.0, 1.0)

    support = resize_mask(support_mask, image.size)
    support_binary = (support > 0.5).astype(np.uint8)
    eroded = cv2.erode(support_binary, np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)

    alpha = alpha * support
    alpha = np.where(eroded > 0.5, 1.0, alpha)
    alpha_img = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(0.8))
    alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.minimum(alpha, support)
    alpha = np.where(alpha < 0.02, 0.0, alpha)
    return alpha


def composite_on_background(image: Image.Image, background: Image.Image, support_mask: np.ndarray):
    image = image.convert("RGB").resize(background.size, Image.Resampling.BICUBIC)
    alpha = estimate_alpha_from_white_bg(image, support_mask)

    fg = np.asarray(image, dtype=np.float32) / 255.0
    bg = np.asarray(background.convert("RGB"), dtype=np.float32) / 255.0
    alpha3 = np.clip(alpha[..., None], 1e-3, 1.0)
    fg_decontaminated = np.clip((fg - (1.0 - alpha[..., None])) / alpha3, 0.0, 1.0)
    comp = fg_decontaminated * alpha[..., None] + bg * (1.0 - alpha[..., None])
    return Image.fromarray((comp * 255).astype(np.uint8), mode="RGB")


def copy_or_save_rgb(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    args = parse_args()
    assets_manifest_path = Path(args.assets_manifest).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    assets_manifest = load_json(assets_manifest_path)
    exported_samples = []

    for sample in assets_manifest.get("samples", []):
        split_short = sample.get("split_short") or sample.get("preset") or "na"
        sample_key = sample.get("sample_key") or "sample"
        sample_dir = output_root / "assets" / split_short / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        input_white = load_rgb(sample.get("input_white_export") or sample.get("input_export"))
        gt_white = load_rgb(sample.get("ground_truth_white_export") or sample.get("ground_truth_export"))
        background = load_rgb(sample.get("target_lighting_export")).resize(gt_white.size, Image.Resampling.BICUBIC)
        target = background.copy()

        mask = infer_mask(gt_white, args.foreground_background_threshold)
        if float(mask.max()) <= 1e-6:
            mask = infer_mask(input_white, args.foreground_background_threshold)
        mask = resize_mask(mask, gt_white.size)

        input_composited = composite_on_background(input_white, background, mask)
        gt_composited = composite_on_background(gt_white, background, mask)

        input_white_path = sample_dir / "input_white_bg.png"
        input_composited_path = sample_dir / "input_composited.png"
        gt_white_path = sample_dir / "ground_truth_white_bg.png"
        gt_composited_path = sample_dir / "ground_truth_composited.png"
        target_path = sample_dir / "target_lighting.png"
        background_path = sample_dir / "target_background.png"
        mask_path = sample_dir / "foreground_mask.png"

        copy_or_save_rgb(input_white, input_white_path)
        copy_or_save_rgb(input_composited, input_composited_path)
        copy_or_save_rgb(gt_white, gt_white_path)
        copy_or_save_rgb(gt_composited, gt_composited_path)
        copy_or_save_rgb(target, target_path)
        copy_or_save_rgb(background, background_path)
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)

        mapped = dict(sample)
        mapped["sample_dir"] = str(sample_dir)
        mapped["input_export"] = str(input_white_path)
        mapped["input_white_export"] = str(input_white_path)
        mapped["input_composited_export"] = str(input_composited_path)
        mapped["ground_truth_export"] = str(gt_white_path)
        mapped["ground_truth_white_export"] = str(gt_white_path)
        mapped["ground_truth_composited_export"] = str(gt_composited_path)
        mapped["target_lighting_export"] = str(target_path)
        mapped["background_export"] = str(background_path)
        mapped["foreground_mask_export"] = str(mask_path)

        methods = {}
        for method_name, method_payload in sample.get("methods", {}).items():
            white_path = Path(method_payload.get("white_bg") or method_payload.get("composited"))
            pred_white = load_rgb(white_path)
            pred_comp = composite_on_background(pred_white, background, mask)

            pred_white_path = sample_dir / f"{method_name}_white_bg.png"
            pred_comp_path = sample_dir / f"{method_name}_composited.png"
            copy_or_save_rgb(pred_white, pred_white_path)
            copy_or_save_rgb(pred_comp, pred_comp_path)

            methods[method_name] = dict(method_payload)
            methods[method_name]["white_bg"] = str(pred_white_path)
            methods[method_name]["composited"] = str(pred_comp_path)

        mapped["methods"] = methods
        exported_samples.append(mapped)

    output_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(assets_manifest_path),
        "samples": exported_samples,
    }
    write_json(output_root / "assets_manifest.json", output_manifest)
    print(f"wrote {output_root / 'assets_manifest.json'}")


if __name__ == "__main__":
    main()
