import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TARGET_IMAGE_PATTERN = re.compile(r"^(\d{3})_(\d{3})_.+\.png$")


def load_object_list(dataset_dir: Path):
    filtered_list = dataset_dir / "filtered_objects.txt"
    if filtered_list.exists():
        return [line.strip() for line in filtered_list.read_text().splitlines() if line.strip()]
    training_list = dataset_dir / "images" / "training_object_list.json"
    if training_list.exists():
        return json.loads(training_list.read_text())
    images_dir = dataset_dir / "images"
    return sorted([p.name for p in images_dir.iterdir() if p.is_dir()])


def collect_target_images(object_dir: Path):
    all_png = list(object_dir.glob("*.png"))
    return [p for p in all_png if TARGET_IMAGE_PATTERN.match(p.name)]


def pick_samples(dataset_dir: Path, dataset_name: str, sample_count: int, rng: random.Random):
    images_root = dataset_dir / "images"
    lighting_root = dataset_dir / "lighting" / "HDR_rescaled"
    object_ids = load_object_list(dataset_dir)
    rng.shuffle(object_ids)

    samples = []
    for obj_id in object_ids:
        if len(samples) >= sample_count:
            break
        object_dir = images_root / obj_id
        if not object_dir.exists():
            continue
        candidates = sorted(collect_target_images(object_dir))
        if not candidates:
            continue
        image_path = candidates[0]
        hdr_path = lighting_root / obj_id / image_path.name
        if not hdr_path.exists():
            alt = sorted((lighting_root / obj_id).glob("*.png"))
            hdr_path = alt[0] if alt else None
        samples.append(
            {
                "dataset": dataset_name,
                "object_id": obj_id,
                "image_path": str(image_path),
                "hdr_path": str(hdr_path) if hdr_path else None,
            }
        )
    return samples


def resolve_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def build_collage(samples, tile_size, columns, padding, label_key, draw_labels=True):
    rows = (len(samples) + columns - 1) // columns
    width = columns * tile_size + (columns - 1) * padding
    height = rows * tile_size + (rows - 1) * padding
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = resolve_font(max(int(tile_size * 0.08), 10))

    for idx, sample in enumerate(samples):
        row = idx // columns
        col = idx % columns
        x = col * (tile_size + padding)
        y = row * (tile_size + padding)
        img = Image.open(sample[label_key]).convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
        canvas.paste(img, (x, y))
        if draw_labels:
            label = sample["dataset"]
            if font:
                draw.text((x + 6, y + 6), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            else:
                draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
    return canvas


def build_target_lighting_panel(samples_by_dataset, tile_size, padding, header_text=None, draw_header=True, draw_labels=False):
    dataset_names = list(samples_by_dataset.keys())
    strip_count = max(len(samples_by_dataset[name]) for name in dataset_names)
    strip_width = strip_count * tile_size + (strip_count - 1) * padding
    strip_height = tile_size

    header_font = resolve_font(max(int(tile_size * 0.18), 18))
    label_font = resolve_font(max(int(tile_size * 0.08), 10))

    header_height = max(int(tile_size * 0.4), 48)
    inter_section = max(int(tile_size * 0.25), 32)
    section_gap = max(int(tile_size * 0.3), 40)

    total_height = header_height + len(dataset_names) * strip_height + (len(dataset_names) - 1) * section_gap
    canvas = Image.new("RGB", (strip_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if draw_header and header_text and header_font:
        draw.text((0, int(header_height * 0.2)), header_text, fill=(0, 0, 0), font=header_font)

    y = header_height
    for dataset_name in dataset_names:
        samples = samples_by_dataset[dataset_name]
        for idx in range(strip_count):
            x = idx * (tile_size + padding)
            if idx >= len(samples):
                fill = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
                canvas.paste(fill, (x, y))
                continue
            sample = samples[idx]
            img = Image.open(sample["hdr_path"]).convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
            canvas.paste(img, (x, y))
        if draw_labels and label_font:
            draw.text((0, y + strip_height + 4), dataset_name, fill=(80, 80, 80), font=label_font)
        y += strip_height + section_gap

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Create showcase collages for ecommerce and 3D-FUTURE datasets.")
    parser.add_argument("--ready_root", type=str, default="/4T/CXY/Neural_Gaffer/logs/ready_subdatasets_20260328")
    parser.add_argument("--datasets", nargs="+", default=["ecommerce", "three_future"])
    parser.add_argument("--sample_per_dataset", type=int, default=6)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--layout", type=str, default="grid", choices=["grid", "target_lighting"])
    parser.add_argument("--no_text", action="store_true", help="Disable all text overlays.")
    parser.add_argument("--rows", type=int, default=None, help="Force number of rows for grid layout.")
    args = parser.parse_args()

    ready_root = Path(args.ready_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (ready_root.parent / f"dataset_showcase_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_samples = []
    for dataset_name in args.datasets:
        dataset_dir = ready_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
        samples = pick_samples(dataset_dir, dataset_name, args.sample_per_dataset, rng)
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("No samples found to build collages.")

    total_samples = len(all_samples)
    if args.rows:
        columns = max(1, (total_samples + args.rows - 1) // args.rows)
    else:
        columns = args.sample_per_dataset
    object_collage = build_collage(
        all_samples,
        tile_size=args.tile_size,
        columns=columns,
        padding=args.padding,
        label_key="image_path",
        draw_labels=not args.no_text,
    )
    object_path = output_dir / "showcase_objects.png"
    object_collage.save(object_path)

    hdr_samples = [s for s in all_samples if s.get("hdr_path")]
    hdr_path = output_dir / "showcase_hdr_lighting.png"
    if args.layout == "target_lighting":
        samples_by_dataset = {}
        for dataset_name in args.datasets:
            samples_by_dataset[dataset_name] = [
                s for s in hdr_samples if s["dataset"] == dataset_name
            ]
        hdr_collage = build_target_lighting_panel(
            samples_by_dataset,
            tile_size=args.tile_size,
            padding=args.padding,
            header_text=None if args.no_text else "Target Lighting",
            draw_header=not args.no_text,
            draw_labels=False,
        )
        hdr_path = output_dir / "showcase_hdr_target_lighting.png"
        hdr_collage.save(hdr_path)
    else:
        hdr_collage = build_collage(
            hdr_samples,
            tile_size=args.tile_size,
            columns=columns,
            padding=args.padding,
            label_key="hdr_path",
            draw_labels=not args.no_text,
        )
        hdr_collage.save(hdr_path)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_root": str(ready_root),
        "datasets": args.datasets,
        "sample_per_dataset": args.sample_per_dataset,
        "tile_size": args.tile_size,
        "padding": args.padding,
        "seed": args.seed,
        "object_collage": str(object_path),
        "hdr_collage": str(hdr_path),
        "samples": all_samples,
    }
    (output_dir / "showcase_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote object collage: {object_path}")
    print(f"Wrote HDR collage: {hdr_path}")
    print(f"Wrote meta: {output_dir / 'showcase_meta.json'}")


if __name__ == "__main__":
    main()
