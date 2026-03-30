import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


SPLITS = [
    ("uu", "official_uu_main_env.png", "official_uu_main_result.png"),
    ("us", "official_us_main_env.png", "official_us_main_result.png"),
    ("ra", "official_ra_main_env.png", "official_ra_main_result.png"),
]


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def rounded_mask(size, radius):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def crop_hero_object(result_image: Image.Image) -> Image.Image:
    header_h = 18
    column_w = result_image.width // 4
    # Use the final column as the main-method prediction.
    obj = result_image.crop((column_w * 3, header_h, result_image.width, result_image.height)).convert("RGBA")
    rgb = obj.convert("RGB")
    inv = ImageChops.invert(rgb)
    bbox = inv.getbbox()
    if bbox is None:
        return obj
    obj = obj.crop(bbox)

    # Treat near-white pixels as transparent background.
    alpha = Image.new("L", obj.size, 255)
    alpha_px = alpha.load()
    rgb_px = obj.convert("RGB").load()
    for y in range(obj.height):
        for x in range(obj.width):
            r, g, b = rgb_px[x, y]
            if min(r, g, b) > 244:
                alpha_px[x, y] = 0
    obj.putalpha(alpha.filter(ImageFilter.GaussianBlur(1.4)))
    return obj


def make_background(env_image: Image.Image, size):
    bg = env_image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(8))
    overlay = Image.new("RGBA", size, (9, 14, 26, 92))
    return Image.alpha_composite(bg.convert("RGBA"), overlay)


def fit_object(obj: Image.Image, card_size):
    target_h = int(card_size[1] * 0.78)
    scale = target_h / max(obj.height, 1)
    new_size = (max(1, int(obj.width * scale)), max(1, int(obj.height * scale)))
    return obj.resize(new_size, Image.Resampling.LANCZOS)


def draw_card(canvas: Image.Image, xy, size, title, env_path: Path, result_path: Path, fonts):
    x, y = xy
    w, h = size
    card = Image.new("RGBA", size, (255, 255, 255, 0))
    env = Image.open(env_path)
    result = Image.open(result_path)

    bg = make_background(env, size)
    card.alpha_composite(bg)

    # Subtle inner glow.
    glow = Image.new("RGBA", size, (255, 255, 255, 0))
    g = ImageDraw.Draw(glow)
    g.rounded_rectangle((10, 10, w - 10, h - 10), radius=28, outline=(255, 255, 255, 70), width=2)
    card.alpha_composite(glow)

    obj = fit_object(crop_hero_object(result), size)
    ox = (w - obj.width) // 2
    oy = h - obj.height - 26

    shadow = Image.new("RGBA", (obj.width + 28, obj.height + 28), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (obj.width, obj.height), (0, 0, 0, 135))
    shadow.alpha_composite(shadow_layer, (14, 14))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    card.alpha_composite(shadow, (ox - 4, oy + 8))
    card.alpha_composite(obj, (ox, oy))

    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((20, 20, 96, 58), radius=18, fill=(255, 255, 255, 210))
    draw.text((38, 31), title, font=fonts["badge"], fill=(25, 33, 53))
    draw.text((24, h - 48), "target-lighting background + relit object", font=fonts["small"], fill=(255, 255, 255))

    card_mask = rounded_mask(size, 34)
    canvas.paste(card, (x, y), card_mask)


def build_teaser(crops_dir: Path, output_path: Path):
    width, height = 1840, 900
    canvas = Image.new("RGBA", (width, height), (244, 246, 250, 255))

    # Large soft background shapes.
    bg = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bg)
    bdraw.ellipse((-140, 520, 520, 1120), fill=(159, 196, 255, 135))
    bdraw.ellipse((1220, -120, 1860, 480), fill=(255, 214, 170, 120))
    bdraw.rounded_rectangle((250, 120, 1590, 820), radius=52, fill=(255, 255, 255, 215))
    bg = bg.filter(ImageFilter.GaussianBlur(22))
    canvas.alpha_composite(bg)

    fonts = {
        "title": load_font(42, bold=True),
        "subtitle": load_font(20, bold=False),
        "badge": load_font(24, bold=True),
        "small": load_font(20, bold=False),
    }

    draw = ImageDraw.Draw(canvas)
    draw.text((84, 72), "Official Held-out Relighting Showcase", font=fonts["title"], fill=(19, 29, 48))
    draw.text((86, 128), "Each panel places the relit object back into a lighting-aware background, closer to the Neural Gaffer website presentation style.", font=fonts["subtitle"], fill=(84, 97, 123))

    card_w, card_h = 520, 620
    gap = 44
    start_x = 88
    y = 200
    for idx, (title, env_name, result_name) in enumerate(SPLITS):
        x = start_x + idx * (card_w + gap)
        draw_card(
            canvas,
            (x, y),
            (card_w, card_h),
            title.upper(),
            crops_dir / env_name,
            crops_dir / result_name,
            fonts,
        )

    footer = "Suggested paper usage: place this teaser before the detailed crop comparison figure."
    draw.text((86, 850), footer, font=fonts["subtitle"], fill=(100, 110, 132))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Generate a website-style background showcase figure for the paper.")
    parser.add_argument("--crops_dir", type=str, default="docs/figures/crops")
    parser.add_argument("--output", type=str, default="docs/figures/official_background_teaser.png")
    args = parser.parse_args()

    build_teaser(Path(args.crops_dir), Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
