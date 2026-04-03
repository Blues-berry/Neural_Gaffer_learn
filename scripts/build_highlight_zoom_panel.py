import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Build a highlight-focused zoom panel from exported relighting assets.")
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help=(
            "Explicit column order. Supported tokens: input_image, ground_truth, target_lighting, "
            "foreground_mask, gt_highlight_mask, method:<name>, method_mask:<name>."
        ),
    )
    parser.add_argument("--methods", nargs="*", default=["baseline", "ours_full"])
    parser.add_argument("--focus-methods", nargs="*", default=None, help="Methods whose highlight masks are used when computing the zoom bbox.")
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=72)
    parser.add_argument("--crop-padding", type=int, default=20)
    parser.add_argument("--min-crop-size", type=int, default=72)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def pretty_column_name(name: str):
    mapping = {
        "baseline": "Baseline (Gaffer)",
        "ours_full": "Ours (Full)",
        "official-demo": "Official Demo",
        "ground-truth": "Ground-truth",
        "target lighting": "Target Lighting",
        "input image": "Input Image",
        "foreground mask": "Foreground Mask",
        "gt highlight mask": "GT M_h",
    }
    return mapping.get(name.lower(), name)


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
        if (bbox[2] - bbox[0]) <= max_width - 8:
            return font
    return load_font(min_size, bold=bold)


def centered_text(draw: ImageDraw.ImageDraw, box, text, font, fill):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def open_or_placeholder(path: str | None, size, label: str):
    if path and Path(path).exists():
        return Image.open(path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    image = Image.new("RGB", size, (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, size[0] - 8, size[1] - 8), radius=18, outline=(188, 194, 205), width=2)
    font = load_font(max(18, size[0] // 12), bold=False)
    centered_text(draw, (0, 0, size[0], size[1]), label, font, (100, 108, 122))
    return image


def resolve_column(sample: dict, token: str):
    token = token.strip()
    token_lower = token.lower()
    if token_lower in {"input", "input_image"}:
        return pretty_column_name("Input Image"), sample.get("input_export")
    if token_lower in {"ground_truth", "gt"}:
        return pretty_column_name("Ground-truth"), sample.get("ground_truth_export")
    if token_lower in {"target_lighting", "lighting"}:
        return pretty_column_name("Target Lighting"), sample.get("target_lighting_export")
    if token_lower in {"foreground_mask", "mask"}:
        return pretty_column_name("Foreground Mask"), sample.get("foreground_mask_export")
    if token_lower in {"gt_highlight_mask", "gt_mh", "mh"}:
        return pretty_column_name("GT Highlight Mask"), sample.get("gt_highlight_mask_export")
    if token_lower.startswith("method:"):
        method_name = token.split(":", 1)[1]
        return pretty_column_name(method_name), sample.get("methods", {}).get(method_name, {}).get("composited")
    if token_lower.startswith("method_mask:"):
        method_name = token.split(":", 1)[1]
        return f"{pretty_column_name(method_name)} M_h", sample.get("methods", {}).get(method_name, {}).get("highlight_mask")
    raise ValueError(f"Unsupported column token: {token}")


def load_mask(path: str | None):
    if not path:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        return None
    image = Image.open(mask_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, target_size: tuple[int, int]):
    target_w, target_h = target_size
    image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def square_bbox_from_mask(mask: np.ndarray, padding: int = 0, min_crop_size: int = 72):
    ys, xs = np.nonzero(mask > 0.5)
    height, width = mask.shape
    if ys.size == 0 or xs.size == 0:
        side = min(max(int(min_crop_size), min(height, width) // 2), min(height, width))
        cy = height / 2.0
        cx = width / 2.0
    else:
        top = int(ys.min()) - int(padding)
        bottom = int(ys.max()) + int(padding) + 1
        left = int(xs.min()) - int(padding)
        right = int(xs.max()) + int(padding) + 1
        cy = (top + bottom) / 2.0
        cx = (left + right) / 2.0
        side = max(bottom - top, right - left, int(min_crop_size))

    side = min(int(side), min(height, width))
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


def crop_image(image: Image.Image, bbox):
    if bbox is None:
        return image
    top, bottom, left, right = bbox
    return image.crop((left, top, right, bottom))


def build_focus_mask(sample: dict, focus_methods: list[str] | None):
    gt_mask = load_mask(sample.get("gt_highlight_mask_binary_export"))
    if gt_mask is None:
        gt_mask = load_mask(sample.get("foreground_mask_export"))
    if gt_mask is None:
        return None

    focus_mask = np.asarray(gt_mask, dtype=np.float32)
    method_names = focus_methods or sorted(sample.get("methods", {}).keys())
    for method_name in method_names:
        method_mask = load_mask(sample.get("methods", {}).get(method_name, {}).get("highlight_mask_binary"))
        if method_mask is None:
            continue
        if method_mask.shape != focus_mask.shape:
            method_mask = resize_mask(method_mask, (focus_mask.shape[1], focus_mask.shape[0]))
        focus_mask = np.clip(focus_mask + method_mask, 0.0, 1.0)
    return focus_mask


def main():
    args = parse_args()
    assets = json.loads(Path(args.assets_manifest).read_text(encoding="utf-8"))
    samples = list(assets.get("samples", []))
    if args.max_samples is not None:
        samples = samples[: max(int(args.max_samples), 0)]

    if args.columns:
        column_tokens = list(args.columns)
    else:
        column_tokens = [
            "method:baseline",
            "method:ours_full",
            "ground_truth",
            "gt_highlight_mask",
            "method_mask:baseline",
            "method_mask:ours_full",
        ]

    column_labels = [resolve_column(samples[0], token)[0] for token in column_tokens] if samples else []
    cols = len(column_labels)
    rows = len(samples)

    tile_w = args.tile_size
    tile_h = args.tile_size
    left_margin = 28
    top_margin = 20
    width = left_margin * 2 + cols * tile_w + (cols - 1) * args.padding
    height = top_margin * 2 + args.header_height + rows * tile_h + (rows - 1) * args.padding

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    row_font = load_font(18, bold=False)

    x = left_margin
    for col_name in column_labels:
        header_font = fit_font(col_name, tile_w, start_size=28, min_size=16, bold=False)
        centered_text(draw, (x, top_margin, x + tile_w, top_margin + args.header_height - 16), col_name, header_font, (24, 24, 28))
        x += tile_w + args.padding

    y = top_margin + args.header_height
    for sample in samples:
        focus_mask = build_focus_mask(sample, args.focus_methods)
        bbox = square_bbox_from_mask(focus_mask, padding=args.crop_padding, min_crop_size=args.min_crop_size) if focus_mask is not None else None

        x = left_margin
        for token, col_name in zip(column_tokens, column_labels):
            _, image_path = resolve_column(sample, token)
            tile = open_or_placeholder(image_path, (tile_w, tile_h), f"{col_name}\npending")
            tile = crop_image(tile, bbox).resize((tile_w, tile_h), Image.Resampling.BICUBIC)
            canvas.paste(tile, (x, y))
            x += tile_w + args.padding

        sample_label = sample.get("display_name") or f"{sample['preset'].upper()}  {sample['object_id'][:8]}"
        draw.text((left_margin, y + tile_h + 4), sample_label, font=row_font, fill=(138, 144, 156))
        y += tile_h + args.padding

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
