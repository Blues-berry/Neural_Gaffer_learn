import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_REPO = Path("/4T/CXY/Neural_Gaffer_original_main_baseline")
DEFAULT_CHECKPOINT_ROOT = Path("/4T/CXY/Neural_Gaffer_original/logs/neural_gaffer_res256")
DEFAULT_ZERO123_PATH = Path(
    "/4T/huggingface_cache/models--kxic--zero123-xl/snapshots/7d8aec2223b93e84eb26893d1e732e013523474b"
)
RUNTIME_SHIM_DIR = REPO_ROOT / "scripts" / "runtime_shims"

DEFAULT_IMAGES = [
    "dragon.jpg",
    "Mandalorian_helmet.jpg",
    "duck.png",
]

DEFAULT_ENVMAPS = [
    "012_hdrmaps_com_free_2K.exr",
    "064_hdrmaps_com_free_2K.exr",
    "117_hdrmaps_com_free_2K.exr",
    "128_hdrmaps_com_free_2K.exr",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the official Neural Gaffer demo on curated demo-folder assets.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--artifact-prefix", default="official_demo_gallery")
    parser.add_argument("--demo-dir", default="demo")
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    parser.add_argument("--envmaps", nargs="*", default=DEFAULT_ENVMAPS)
    parser.add_argument("--official-repo", default=str(DEFAULT_OFFICIAL_REPO))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--checkpoint-name", default="checkpoint-80000")
    parser.add_argument("--pretrained-model-name-or-path", default=str(DEFAULT_ZERO123_PATH))
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-validation-images", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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


def centered_text(draw: ImageDraw.ImageDraw, box, text, font, fill):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def pretty_lighting_label(name: str):
    token = name.split("_")[0]
    if token.isdigit():
        return f"HDRI {token}"
    return name.replace("_", " ")


def open_or_placeholder(path: Path, size, label: str):
    if path.exists():
        return Image.open(path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    image = Image.new("RGB", size, (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, size[0] - 8, size[1] - 8), radius=18, outline=(188, 194, 205), width=2)
    centered_text(draw, (0, 0, size[0], size[1]), label, load_font(18, bold=False), (100, 108, 122))
    return image


def alpha_from_image(image: Image.Image):
    if "A" not in image.getbands():
        return None
    alpha = np.array(image.getchannel("A"), dtype=np.uint8)
    if alpha.min() >= 250:
        return None
    return alpha


def build_grabcut_mask(image: Image.Image, stem: str):
    rgb = np.array(image.convert("RGB").resize((768, 768), Image.Resampling.LANCZOS))
    h, w = rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    margin = 24
    mask[:margin, :] = cv2.GC_BGD
    mask[-margin:, :] = cv2.GC_BGD
    mask[:, :margin] = cv2.GC_BGD
    mask[:, -margin:] = cv2.GC_BGD

    y_grid, x_grid = np.ogrid[:h, :w]
    cx = w // 2
    cy = h // 2

    if stem == "dragon":
        sure_fg = ((x_grid - cx) ** 2) / (230**2) + ((y_grid - (cy + 20)) ** 2) / (210**2) <= 1.0
        likely_fg = ((x_grid - cx) ** 2) / (320**2) + ((y_grid - (cy + 10)) ** 2) / (250**2) <= 1.0
        mask[int(h * 0.78) :, :] = cv2.GC_BGD
        bottom_center = np.s_[int(h * 0.62) :, int(w * 0.25) : int(w * 0.75)]
        mask[bottom_center] = np.where(mask[bottom_center] == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_PR_FGD)
    elif stem == "vege_dog":
        sure_fg = ((x_grid - cx) ** 2) / (210**2) + ((y_grid - (cy + 20)) ** 2) / (250**2) <= 1.0
        likely_fg = ((x_grid - cx) ** 2) / (300**2) + ((y_grid - (cy + 20)) ** 2) / (330**2) <= 1.0
    elif stem == "statue_of_hand":
        sure_fg = ((x_grid - cx) ** 2) / (110**2) + ((y_grid - (cy + 40)) ** 2) / (330**2) <= 1.0
        likely_fg = ((x_grid - cx) ** 2) / (180**2) + ((y_grid - (cy + 40)) ** 2) / (430**2) <= 1.0
        mask[int(h * 0.86) :, :] = cv2.GC_BGD
    else:
        sure_fg = ((x_grid - cx) ** 2) / (190**2) + ((y_grid - cy) ** 2) / (220**2) <= 1.0
        likely_fg = ((x_grid - cx) ** 2) / (280**2) + ((y_grid - cy) ** 2) / (320**2) <= 1.0

    mask[sure_fg] = cv2.GC_FGD
    mask[likely_fg] = np.where(mask[likely_fg] == cv2.GC_BGD, cv2.GC_BGD, cv2.GC_PR_FGD)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, None, bg_model, fg_model, 6, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        centers = centroids[1:]
        center_score = areas - 0.25 * np.sum((centers - [w / 2.0, h / 2.0]) ** 2, axis=1)
        keep_idx = 1 + int(center_score.argmax())
        mask = np.where(labels == keep_idx, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = Image.fromarray(mask).resize(image.size, Image.Resampling.LANCZOS)
    return np.array(mask, dtype=np.uint8)


def prepare_rgba(image_path: Path):
    image = Image.open(image_path)
    alpha = alpha_from_image(image)
    if alpha is None:
        alpha = build_grabcut_mask(image, image_path.stem)
        rgba = np.dstack([np.array(image.convert("RGB"), dtype=np.uint8), alpha])
        return Image.fromarray(rgba, mode="RGBA")
    return image.convert("RGBA")


def recenter_rgba_to_official_input(rgba: Image.Image):
    image_arr = np.array(rgba)
    alpha = image_arr[..., 3]
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    x, y, width, height = cv2.boundingRect(binary)
    max_size = max(width, height)
    side_len = max(1, int(max_size / 0.75))
    padded = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded[center - height // 2 : center - height // 2 + height, center - width // 2 : center - width // 2 + width] = image_arr[
        y : y + height,
        x : x + width,
    ]
    rgba_square = Image.fromarray(padded, mode="RGBA").resize((256, 256), Image.Resampling.LANCZOS)
    rgba_arr = np.array(rgba_square).astype(np.float32) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., 3:4] + (1.0 - rgba_arr[..., 3:4])
    mask = rgba_arr[..., 3]
    return (
        Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB"),
        Image.fromarray((mask * 255.0).astype(np.uint8), mode="L"),
    )


def preprocess_inputs(demo_dir: Path, image_names, output_root: Path):
    prep_root = output_root / "preprocessed_data"
    img_dir = prep_root / "img"
    mask_dir = prep_root / "mask"
    debug_dir = prep_root / "debug"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for image_name in image_names:
        image_path = demo_dir / image_name
        rgba = prepare_rgba(image_path)
        input_img, mask_img = recenter_rgba_to_official_input(rgba)
        stem = image_path.stem
        input_path = img_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        debug_rgba_path = debug_dir / f"{stem}_rgba.png"
        debug_input_path = debug_dir / f"{stem}_input.png"
        input_img.save(input_path)
        mask_img.save(mask_path)
        rgba.save(debug_rgba_path)
        input_img.save(debug_input_path)
        manifest.append(
            {
                "image_name": image_name,
                "stem": stem,
                "original_path": str(image_path),
                "preprocessed_input": str(input_path),
                "mask_path": str(mask_path),
                "debug_rgba_path": str(debug_rgba_path),
            }
        )
    return img_dir, mask_dir, manifest


def prepare_envmaps(demo_dir: Path, envmaps, output_root: Path):
    selected_dir = output_root / "selected_envmaps"
    lighting_dir = output_root / "preprocessed_lighting_data"
    selected_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for envmap_name in envmaps:
        src = demo_dir / "environment_map_sample" / envmap_name
        if not src.exists():
            src = demo_dir / "hdrmaps_for_3d" / envmap_name
        if not src.exists():
            raise FileNotFoundError(f"Environment map not found in demo folder: {envmap_name}")
        dst = selected_dir / envmap_name
        shutil.copy2(src, dst)
        copied.append(dst)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_bg_and_rotate_envir_map.py"),
        "--lighting_dir",
        str(selected_dir),
        "--output_dir",
        str(lighting_dir),
        "--frame_num",
        "1",
        "--init_RT_path",
        str(demo_dir / "default_pose.npy"),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return lighting_dir, [path.stem for path in copied]


def run_official_inference(args, output_root: Path, prep_img_dir: Path, lighting_dir: Path):
    raw_output_dir = output_root / "raw_outputs"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    official_repo = resolve_path(args.official_repo)
    checkpoint_root = resolve_path(args.checkpoint_root)
    pretrained_path = resolve_path(args.pretrained_model_name_or_path)
    log_path = output_root / f"{args.artifact_prefix}.log"

    cmd = [
        sys.executable,
        str(official_repo / "neural_gaffer_inference_real_data.py"),
        "--pretrained_model_name_or_path",
        str(pretrained_path),
        "--output_dir",
        str(checkpoint_root),
        "--mixed_precision",
        args.mixed_precision,
        "--resume_from_checkpoint",
        args.checkpoint_name,
        "--total_view",
        "1",
        "--lighting_per_view",
        str(len(args.envmaps)),
        "--val_img_dir",
        str(prep_img_dir),
        "--val_lighting_dir",
        str(lighting_dir),
        "--save_dir",
        str(raw_output_dir),
        "--seed",
        str(args.seed),
        "--num_validation_images",
        str(args.num_validation_images),
        "--enable_xformers_memory_efficient_attention",
        "false",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    env["PYTHONPATH"] = (
        f"{RUNTIME_SHIM_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(RUNTIME_SHIM_DIR)
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, cwd=official_repo, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    return raw_output_dir, log_path


def composite_backgrounds(mask_dir: Path, lighting_dir: Path, raw_output_dir: Path, output_root: Path):
    composited_dir = output_root / "composited"
    composited_dir.mkdir(parents=True, exist_ok=True)

    object_names = [path.name for path in raw_output_dir.iterdir() if path.is_dir()]
    lighting_names = [path.name for path in lighting_dir.iterdir() if path.is_dir()]

    for object_name in object_names:
        mask = np.array(Image.open(mask_dir / f"{object_name}.png").convert("L"), dtype=np.float32) / 255.0
        blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
        eroded_mask = cv2.erode(blurred_mask, np.ones((3, 3), np.uint8), iterations=1)[..., np.newaxis]
        pred_dir = raw_output_dir / object_name / "pred_image"

        for lighting_name in lighting_names:
            output_dir = composited_dir / lighting_name / object_name
            output_dir.mkdir(parents=True, exist_ok=True)

            pred_paths = sorted(pred_dir.glob(f"{lighting_name}_*.png"))
            for pred_path in pred_paths:
                frame_token = pred_path.stem.split("_")[-1]
                bg_path = lighting_dir / lighting_name / "background" / f"{int(frame_token)}.png"
                pred = np.array(Image.open(pred_path).convert("RGB"), dtype=np.float32) / 255.0
                bg = np.array(Image.open(bg_path).convert("RGB"), dtype=np.float32) / 255.0
                comp = pred * eroded_mask + bg * (1.0 - eroded_mask)
                Image.fromarray((comp * 255.0).astype(np.uint8), mode="RGB").save(output_dir / f"{int(frame_token):03d}.png")

    return composited_dir


def build_panel(output_root: Path, image_entries, lighting_names, tile_size: int):
    panel_path = output_root / "official_demo_gallery_panel_v1.png"
    rows = len(image_entries)
    cols = 1 + len(lighting_names)
    padding = 18
    left_margin = 28
    top_margin = 20
    header_height = 132
    width = left_margin * 2 + cols * tile_size + (cols - 1) * padding
    height = top_margin * 2 + header_height + rows * tile_size + (rows - 1) * padding

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=False)
    label_font = load_font(18, bold=False)
    small_font = load_font(16, bold=False)

    headers = ["Input Image"] + [pretty_lighting_label(name) for name in lighting_names]
    x = left_margin
    for idx, header in enumerate(headers):
        centered_text(draw, (x, top_margin, x + tile_size, top_margin + 36), header, title_font, (24, 24, 28))
        if idx > 0:
            lighting_name = lighting_names[idx - 1]
            thumb_path = output_root / "preprocessed_lighting_data" / lighting_name / "LDR" / "0.png"
            thumb = open_or_placeholder(thumb_path, (min(140, tile_size - 40), 64), header)
            thumb_x = x + (tile_size - thumb.width) // 2
            thumb_y = top_margin + 48
            canvas.paste(thumb, (thumb_x, thumb_y))
        x += tile_size + padding

    y = top_margin + header_height
    for entry in image_entries:
        x = left_margin
        input_tile = open_or_placeholder(Path(entry["original_path"]), (tile_size, tile_size), entry["stem"])
        canvas.paste(input_tile, (x, y))
        draw.text((x, y + tile_size + 4), entry["stem"], font=small_font, fill=(138, 144, 156))
        x += tile_size + padding

        for lighting_name in lighting_names:
            pred_path = output_root / "composited" / lighting_name / entry["stem"] / "000.png"
            tile = open_or_placeholder(pred_path, (tile_size, tile_size), f"{lighting_name}\npending")
            canvas.paste(tile, (x, y))
            x += tile_size + padding

        y += tile_size + padding

    canvas.save(panel_path)
    return panel_path


def main():
    args = parse_args()
    output_root = resolve_path(args.output_root)
    demo_dir = resolve_path(args.demo_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prep_img_dir, mask_dir, image_entries = preprocess_inputs(demo_dir, args.images, output_root)
    lighting_dir, lighting_names = prepare_envmaps(demo_dir, args.envmaps, output_root)

    raw_output_dir = output_root / "raw_outputs"
    log_path = output_root / f"{args.artifact_prefix}.log"
    if not (args.skip_existing and raw_output_dir.exists() and any(raw_output_dir.iterdir())):
        raw_output_dir, log_path = run_official_inference(args, output_root, prep_img_dir, lighting_dir)

    composited_dir = composite_backgrounds(mask_dir, lighting_dir, raw_output_dir, output_root)
    panel_path = build_panel(output_root, image_entries, lighting_names, args.tile_size)

    resolved_panel_path = panel_path
    if args.artifact_prefix != "official_demo_gallery":
        resolved_panel_path = output_root / f"{args.artifact_prefix}_panel_v1.png"

    summary = {
        "output_root": str(output_root),
        "images": image_entries,
        "envmaps": args.envmaps,
        "lighting_names": lighting_names,
        "preprocessed_img_dir": str(prep_img_dir),
        "mask_dir": str(mask_dir),
        "lighting_dir": str(lighting_dir),
        "raw_output_dir": str(raw_output_dir),
        "composited_dir": str(composited_dir),
        "panel_path": str(resolved_panel_path),
        "log_path": str(log_path),
        "artifact_prefix": args.artifact_prefix,
    }
    summary_path = output_root / f"{args.artifact_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.artifact_prefix != "official_demo_gallery":
        prefixed_panel_path = output_root / f"{args.artifact_prefix}_panel_v1.png"
        shutil.copy2(panel_path, prefixed_panel_path)
        print(f"wrote {prefixed_panel_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {panel_path}")


if __name__ == "__main__":
    main()
