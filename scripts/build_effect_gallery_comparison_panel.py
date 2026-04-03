import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Build a clean comparison panel from two local real-image relighting galleries.")
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--ours-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=84)
    parser.add_argument("--hide-row-labels", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--input-label", default="Input Image")
    parser.add_argument("--baseline-label", default="Baseline (Gaffer)")
    parser.add_argument("--ours-label", default="Ours (Full)")
    parser.add_argument("--target-label", default="Target Lighting")
    return parser.parse_args()


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


def open_or_placeholder(path: Path, size, label: str):
    if path.exists():
        return Image.open(path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    image = Image.new("RGB", size, (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, size[0] - 8, size[1] - 8), radius=18, outline=(188, 194, 205), width=2)
    centered_text(draw, (0, 0, size[0], size[1]), label, load_font(18, bold=False), (100, 108, 122))
    return image


def gallery_root_from_summary(summary_path: Path):
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    output_root = Path(payload["output_root"])
    images = [entry["stem"] for entry in payload["images"]]
    lighting_names = list(payload["lighting_names"])
    return output_root, images, lighting_names


def main():
    args = parse_args()
    ours_summary = Path(args.ours_summary)

    ours_root, ours_images, ours_lighting = gallery_root_from_summary(ours_summary)
    baseline_root = None
    baseline_images = []
    baseline_lighting = []
    include_baseline = bool(args.baseline_summary)
    if include_baseline:
        baseline_summary = Path(args.baseline_summary)
        baseline_root, baseline_images, baseline_lighting = gallery_root_from_summary(baseline_summary)

    if include_baseline:
        image_names = [name for name in ours_images if name in set(baseline_images)]
        lighting_names = [name for name in ours_lighting if name in set(baseline_lighting)]
    else:
        image_names = list(ours_images)
        lighting_names = list(ours_lighting)

    rows = []
    for image_name in image_names:
        for lighting_name in lighting_names:
            row = {
                "image_name": image_name,
                "lighting_name": lighting_name,
                "input_path": ours_root / "raw_outputs" / image_name / "input_image" / f"{lighting_name}_000.png",
                "ours_path": ours_root / "composited" / lighting_name / image_name / "000.png",
                "target_path": ours_root / "raw_outputs" / image_name / "target_envmap_ldr" / f"{lighting_name}_000.png",
            }
            if include_baseline and baseline_root is not None:
                row["baseline_path"] = baseline_root / "composited" / lighting_name / image_name / "000.png"
            rows.append(row)

    if args.max_rows is not None:
        rows = rows[: max(int(args.max_rows), 0)]

    column_labels = [args.input_label]
    if include_baseline:
        column_labels.append(args.baseline_label)
    column_labels.extend([args.ours_label, args.target_label])
    tile_w = args.tile_size
    tile_h = args.tile_size
    cols = len(column_labels)
    left_margin = 28
    top_margin = 20
    width = left_margin * 2 + cols * tile_w + (cols - 1) * args.padding
    height = top_margin * 2 + args.header_height + len(rows) * tile_h + (len(rows) - 1) * args.padding
    if not args.hide_row_labels:
        height += 24 * len(rows)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    row_font = load_font(18, bold=False)

    x = left_margin
    for col_name in column_labels:
        header_font = fit_font(col_name, tile_w, start_size=28, min_size=16, bold=False)
        centered_text(draw, (x, top_margin, x + tile_w, top_margin + args.header_height - 16), col_name, header_font, (24, 24, 28))
        x += tile_w + args.padding

    y = top_margin + args.header_height
    for row in rows:
        tiles = [open_or_placeholder(row["input_path"], (tile_w, tile_h), "input")]
        if include_baseline:
            tiles.append(open_or_placeholder(row["baseline_path"], (tile_w, tile_h), "baseline"))
        tiles.extend(
            [
                open_or_placeholder(row["ours_path"], (tile_w, tile_h), "ours"),
                open_or_placeholder(row["target_path"], (tile_w, tile_h), "target"),
            ]
        )
        x = left_margin
        for tile in tiles:
            canvas.paste(tile, (x, y))
            x += tile_w + args.padding

        if not args.hide_row_labels:
            label = f"{row['image_name']} / {row['lighting_name']}"
            draw.text((left_margin, y + tile_h + 4), label, font=row_font, fill=(138, 144, 156))
            y += tile_h + args.padding + 24
        else:
            y += tile_h + args.padding

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
