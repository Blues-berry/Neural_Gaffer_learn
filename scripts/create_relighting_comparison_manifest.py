import argparse
import json
import random
from pathlib import Path


PRESETS = {
    "uu": {
        "image_split": "unseen_lighting",
        "lighting_split": "unseen_lighting",
        "input_mode": "paired_lighting",
        "cond_lighting_index": 1,
        "target_lighting_index": 0,
    },
    "us": {
        "image_split": "seen_lighting",
        "lighting_split": "seen_lighting",
        "input_mode": "paired_lighting",
        "cond_lighting_index": 1,
        "target_lighting_index": 0,
    },
    "ra": {
        "image_split": "unseen_lighting",
        "lighting_split": "unseen_lighting",
        "input_mode": "random_lighting",
        "cond_lighting_index": None,
        "target_lighting_index": 0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a manifest for relighting comparison panels.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument(
        "--images-root",
        default="validation_data/images/val_rendered_images_resized/validation",
        help="Root containing seen_lighting and unseen_lighting object folders.",
    )
    parser.add_argument(
        "--lighting-root",
        default="validation_data/lighting/val_preprocessed_environment_resized",
        help="Root containing seen_lighting and unseen_lighting lighting folders.",
    )
    parser.add_argument("--object-ids", nargs="*", default=None, help="Explicit object ids to include.")
    parser.add_argument("--sample-count", type=int, default=3, help="Randomly sample this many objects when object-ids is omitted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-idx", type=int, default=0)
    parser.add_argument("--target-lighting-index", type=int, default=None)
    parser.add_argument("--cond-lighting-index", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path. Defaults to logs/relighting_comparison/<preset>_manifest.json",
    )
    return parser.parse_args()


def find_matching_file(object_dir: Path, view_idx: int, lighting_idx: int):
    pattern = f"{view_idx:03d}_{lighting_idx:03d}_*.png"
    matches = sorted(object_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {object_dir}")
    return matches[0]


def build_sample(args, preset_name: str, object_id: str):
    preset = PRESETS[preset_name]
    image_object_dir = Path(args.images_root) / preset["image_split"] / object_id
    lighting_split_dir = Path(args.lighting_root) / preset["lighting_split"]
    target_lighting_index = preset["target_lighting_index"] if args.target_lighting_index is None else args.target_lighting_index
    cond_lighting_index = preset["cond_lighting_index"] if args.cond_lighting_index is None else args.cond_lighting_index

    gt_path = find_matching_file(image_object_dir, args.view_idx, target_lighting_index)
    if preset["input_mode"] == "random_lighting":
        input_path = image_object_dir / f"random_lighting_{args.view_idx:03d}.png"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing random-light input: {input_path}")
    else:
        if cond_lighting_index is None:
            raise ValueError(f"cond_lighting_index is required for preset {preset_name}")
        input_path = find_matching_file(image_object_dir, args.view_idx, cond_lighting_index)

    target_file = gt_path.name
    target_lighting_ldr_path = lighting_split_dir / "LDR" / object_id / target_file
    target_lighting_hdr_path = lighting_split_dir / "HDR_rescaled" / object_id / target_file
    if not target_lighting_hdr_path.exists():
        target_lighting_hdr_path = lighting_split_dir / "HDR_normalized" / object_id / target_file

    sample = {
        "preset": preset_name,
        "object_id": object_id,
        "view_idx": args.view_idx,
        "target_lighting_index": target_lighting_index,
        "cond_lighting_index": cond_lighting_index,
        "image_split": preset["image_split"],
        "lighting_split": preset["lighting_split"],
        "input_mode": preset["input_mode"],
        "target_file": target_file,
        "input_path": str(input_path),
        "gt_path": str(gt_path),
        "target_lighting_ldr_path": str(target_lighting_ldr_path),
        "target_lighting_hdr_path": str(target_lighting_hdr_path),
    }
    return sample


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    image_split_root = Path(args.images_root) / preset["image_split"]
    if args.object_ids:
        object_ids = list(args.object_ids)
    else:
        object_ids = sorted([p.name for p in image_split_root.iterdir() if p.is_dir()])
        rng = random.Random(args.seed)
        rng.shuffle(object_ids)
        object_ids = object_ids[: args.sample_count]

    samples = [build_sample(args, args.preset, object_id) for object_id in object_ids]
    output_path = Path(args.output) if args.output else Path("logs/relighting_comparison") / f"{args.preset}_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": args.preset,
        "view_idx": args.view_idx,
        "sample_count": len(samples),
        "samples": samples,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
