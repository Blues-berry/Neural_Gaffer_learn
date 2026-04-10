import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Build a relighting comparison panel from exported assets.")
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
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["dilightnet", "ours"],
        help="Method column order to render before ground-truth and target lighting.",
    )
    parser.add_argument(
        "--method-image-key",
        choices=["composited", "white_bg"],
        default="composited",
        help="Which exported method image to render for method columns.",
    )
    parser.add_argument(
        "--input-image-key",
        choices=["white", "composited"],
        default="white",
        help="Which exported input image variant to render for the input column.",
    )
    parser.add_argument(
        "--ground-truth-image-key",
        choices=["white", "composited"],
        default="white",
        help="Which exported ground-truth image variant to render for the ground-truth column.",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=72)
    parser.add_argument(
        "--preserve-native-size",
        action="store_true",
        help="Preserve each source image's native resolution when stitching instead of resizing to a fixed tile size.",
    )
    parser.add_argument("--hide-headers", action="store_true")
    parser.add_argument("--hide-row-labels", action="store_true")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Hide both headers and row labels to create a clean image-only panel.",
    )
    return parser.parse_args()


def pretty_column_name(name: str):
    mapping = {
        "dilightnet": "DiLightNet",
        "rgbx": "RGB<->X",
        "ours": "Ours",
        "ours_full": "Ours (Full)",
        "officialval": "Ours (OfficialVal)",
        "baseline_0316_fallback": "0316 Baseline",
        "jbhdfvfc_ckpt80k": "80K Highlight",
        "cosine0331_03": "Cosine 0331-03",
        "xkmlb19f_like_relative_fallback": "Relative Fallback",
        "hyblite_0331_02_fallback": "Abl. Hybrid Lite",
        "officialval_0403_04": "Ours (OfficialVal)",
        "abl00_base": "Abl. Base",
        "abl01_imgspace_fixed": "Abl. ImgSpace Fixed",
        "abl02_quantile": "Abl. Quantile",
        "abl03_blur": "Abl. Blur",
        "abl04_relative": "Abl. Relative",
        "abl05_full_main": "Abl. Full Main",
        "hyblite": "Abl. Hybrid Lite",
        "freqsplit": "Abl. Freq Split",
        "cosine_lowlr": "Abl. Cosine LowLR",
        "baseline": "Neural Gaffer",
        "7cn19b1e": "Neural Gaffer",
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


def load_native_size(path: str | None, fallback_size):
    if path and Path(path).exists():
        with Image.open(path) as image:
            return image.size
    return fallback_size


def open_native_or_placeholder(path: str | None, size, label: str):
    if path and Path(path).exists():
        return Image.open(path).convert("RGB")
    return open_or_placeholder(None, size, label)


def resolve_column(sample: dict, token: str, method_image_key: str, input_image_key: str, ground_truth_image_key: str):
    token = token.strip()
    token_lower = token.lower()
    if token_lower in {"input", "input_image"}:
        input_path = (
            sample.get("input_composited_export")
            if input_image_key == "composited"
            else sample.get("input_white_export") or sample.get("input_export")
        )
        return pretty_column_name("Input Image"), input_path
    if token_lower in {"ground_truth", "gt"}:
        gt_path = (
            sample.get("ground_truth_composited_export")
            if ground_truth_image_key == "composited"
            else sample.get("ground_truth_white_export") or sample.get("ground_truth_export")
        )
        return pretty_column_name("Ground-truth"), gt_path
    if token_lower in {"target_lighting", "lighting"}:
        return pretty_column_name("Target Lighting"), sample.get("target_lighting_export")
    if token_lower in {"foreground_mask", "mask"}:
        return pretty_column_name("Foreground Mask"), sample.get("foreground_mask_export")
    if token_lower in {"gt_highlight_mask", "gt_mh", "mh"}:
        return pretty_column_name("GT Highlight Mask"), sample.get("gt_highlight_mask_export")
    if token_lower.startswith("method:"):
        method_name = token.split(":", 1)[1]
        method_entry = sample.get("methods", {}).get(method_name, {})
        method_path = method_entry.get(method_image_key) or method_entry.get("composited") or method_entry.get("white_bg")
        return pretty_column_name(method_name), method_path
    if token_lower.startswith("method_mask:"):
        method_name = token.split(":", 1)[1]
        return f"{pretty_column_name(method_name)} M_h", sample.get("methods", {}).get(method_name, {}).get("highlight_mask")
    raise ValueError(f"Unsupported column token: {token}")


def main():
    args = parse_args()
    hide_headers = args.hide_headers or args.no_text
    hide_row_labels = args.hide_row_labels or args.no_text
    assets = json.loads(Path(args.assets_manifest).read_text(encoding="utf-8"))
    samples = assets["samples"]
    if args.columns:
        column_tokens = list(args.columns)
    else:
        column_tokens = ["input_image"] + [f"method:{name}" for name in args.methods] + ["ground_truth", "target_lighting"]
    column_labels = [
        resolve_column(
            samples[0],
            token,
            args.method_image_key,
            args.input_image_key,
            args.ground_truth_image_key,
        )[0]
        for token in column_tokens
    ] if samples else []
    cols = len(column_labels)
    rows = len(samples)

    default_tile_w = args.tile_size
    default_tile_h = args.tile_size
    left_margin = 28
    top_margin = 20
    header_height = 0 if hide_headers else args.header_height
    row_label_height = 0 if hide_row_labels else 28

    resolved_rows = [
        [
            resolve_column(
                sample,
                token,
                args.method_image_key,
                args.input_image_key,
                args.ground_truth_image_key,
            )
            for token in column_tokens
        ]
        for sample in samples
    ]
    if args.preserve_native_size:
        cell_sizes = [
            [load_native_size(path, (default_tile_w, default_tile_h)) for _, path in resolved_columns]
            for resolved_columns in resolved_rows
        ]
        column_widths = [
            max((cell_sizes[row_idx][col_idx][0] for row_idx in range(rows)), default=default_tile_w)
            for col_idx in range(cols)
        ]
        row_heights = [
            max((cell_sizes[row_idx][col_idx][1] for col_idx in range(cols)), default=default_tile_h)
            for row_idx in range(rows)
        ]
    else:
        column_widths = [default_tile_w] * cols
        row_heights = [default_tile_h] * rows

    width = left_margin * 2 + sum(column_widths) + max(cols - 1, 0) * args.padding
    height = (
        top_margin * 2
        + header_height
        + sum(row_heights)
        + max(rows - 1, 0) * args.padding
        + rows * row_label_height
    )

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    row_font = load_font(18, bold=False)

    if not hide_headers:
        x = left_margin
        for col_idx, col_name in enumerate(column_labels):
            col_width = column_widths[col_idx]
            header_font = fit_font(col_name, col_width, start_size=28, min_size=16, bold=False)
            centered_text(draw, (x, top_margin, x + col_width, top_margin + header_height - 16), col_name, header_font, (24, 24, 28))
            x += col_width + args.padding

    y = top_margin + header_height
    for row_idx, sample in enumerate(samples):
        resolved_columns = resolved_rows[row_idx]
        row_height = row_heights[row_idx]
        x = left_margin
        for col_idx, (col_name, image_path) in enumerate(resolved_columns):
            col_width = column_widths[col_idx]
            if args.preserve_native_size:
                tile = open_native_or_placeholder(image_path, (col_width, row_height), f"{col_name}\npending")
                paste_x = x + (col_width - tile.width) // 2
                paste_y = y + (row_height - tile.height) // 2
                canvas.paste(tile, (paste_x, paste_y))
            else:
                tile = open_or_placeholder(image_path, (col_width, row_height), f"{col_name}\npending")
                canvas.paste(tile, (x, y))
            x += col_width + args.padding

        if not hide_row_labels:
            sample_label = sample.get("display_name") or f"{sample['preset'].upper()}  {sample['object_id'][:8]}"
            draw.text((left_margin, y + row_height + 4), sample_label, font=row_font, fill=(138, 144, 156))
        y += row_height + row_label_height + args.padding

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
