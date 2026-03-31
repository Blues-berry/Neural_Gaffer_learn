import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Build a relighting comparison panel from exported assets.")
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["dilightnet", "ours"],
        help="Method column order to render before ground-truth and target lighting.",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=72)
    return parser.parse_args()


def pretty_column_name(name: str):
    mapping = {
        "dilightnet": "DiLightNet",
        "ours": "Ours",
        "ground-truth": "Ground-truth",
        "target lighting": "Target Lighting",
        "input image": "Input Image",
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


def main():
    args = parse_args()
    assets = json.loads(Path(args.assets_manifest).read_text(encoding="utf-8"))
    samples = assets["samples"]
    columns = [pretty_column_name("Input Image")] + [pretty_column_name(name) for name in args.methods] + [
        pretty_column_name("Ground-truth"),
        pretty_column_name("Target Lighting"),
    ]
    cols = len(columns)
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
    for col_name in columns:
        header_font = fit_font(col_name, tile_w, start_size=28, min_size=16, bold=False)
        centered_text(draw, (x, top_margin, x + tile_w, top_margin + args.header_height - 16), col_name, header_font, (24, 24, 28))
        x += tile_w + args.padding

    y = top_margin + args.header_height
    for sample in samples:
        row_images = [
            sample.get("input_export"),
        ]
        for method_name in args.methods:
            row_images.append(sample.get("methods", {}).get(method_name, {}).get("composited"))
        row_images.append(sample.get("ground_truth_export"))
        row_images.append(sample.get("target_lighting_export"))

        x = left_margin
        for col_name, image_path in zip(columns, row_images):
            tile = open_or_placeholder(image_path, (tile_w, tile_h), f"{col_name}\npending")
            canvas.paste(tile, (x, y))
            x += tile_w + args.padding

        sample_label = f"{sample['preset'].upper()}  {sample['object_id'][:8]}"
        draw.text((left_margin, y + tile_h + 4), sample_label, font=row_font, fill=(138, 144, 156))
        y += tile_h + args.padding

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
