import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_IMAGE_PATTERN = re.compile(r"^(\d{3})_(\d{3})_(.+)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build sharded comparison manifests covering all models and all HDRIs from training/validation unions."
    )
    parser.add_argument(
        "--training-images-root",
        default="logs/dataset_unions/full_current_original_official2000_ecommerce1000_3dfuture_landscape/images",
    )
    parser.add_argument(
        "--training-lighting-root",
        default="logs/dataset_unions/full_current_original_official2000_ecommerce1000_3dfuture_landscape/lighting",
    )
    parser.add_argument(
        "--validation-images-root",
        default="logs/dataset_validation_unions/all_ready_plus_official_20260403/images",
    )
    parser.add_argument(
        "--validation-lighting-root",
        default="logs/dataset_validation_unions/all_ready_plus_official_20260403/lighting",
    )
    parser.add_argument("--include-training", action="store_true", default=True)
    parser.add_argument("--no-include-training", dest="include_training", action="store_false")
    parser.add_argument("--include-validation", action="store_true", default=True)
    parser.add_argument("--no-include-validation", dest="include_validation", action="store_false")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--view-indices", nargs="*", type=int, default=None)
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def infer_dataset_name(object_id: str, plain_name: str):
    if "__" in object_id:
        return object_id.split("__", 1)[0]
    return plain_name


def choose_input_path(object_dir: Path, view_idx: int, target_lighting_index: int):
    random_input = object_dir / f"random_lighting_{view_idx:03d}.png"
    if random_input.exists():
        return random_input, "random_lighting", None

    candidates = []
    for path in sorted(object_dir.glob(f"{view_idx:03d}_*.png")):
        if path.name.endswith("_normals.png"):
            continue
        match = TARGET_IMAGE_PATTERN.match(path.name)
        if match is None:
            continue
        lighting_idx = int(match.group(2))
        if lighting_idx != target_lighting_index:
            candidates.append((lighting_idx, path))
    if candidates:
        cond_lighting_index, input_path = candidates[0]
        return input_path, "paired_lighting", cond_lighting_index
    return None, None, None


def iter_training_samples(images_root: Path, lighting_root: Path, allowed_views: set[int] | None):
    for object_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        object_id = object_dir.name
        dataset_name = infer_dataset_name(object_id, "official_1000_train")
        ldr_dir = lighting_root / "LDR" / object_id
        hdr_dir = lighting_root / "HDR_rescaled" / object_id
        if not ldr_dir.exists():
            continue
        if not hdr_dir.exists():
            hdr_dir = lighting_root / "HDR_normalized" / object_id
        if not hdr_dir.exists():
            continue

        for gt_path in sorted(object_dir.glob("*.png")):
            match = TARGET_IMAGE_PATTERN.match(gt_path.name)
            if match is None:
                continue
            view_idx = int(match.group(1))
            target_lighting_index = int(match.group(2))
            env_name = match.group(3)
            if allowed_views is not None and view_idx not in allowed_views:
                continue
            input_path, input_mode, cond_lighting_index = choose_input_path(object_dir, view_idx, target_lighting_index)
            if input_path is None:
                continue
            target_ldr = ldr_dir / gt_path.name
            target_hdr = hdr_dir / gt_path.name
            if not target_ldr.exists() or not target_hdr.exists():
                continue
            yield {
                "preset": "train_all",
                "source_bucket": "training",
                "dataset": dataset_name,
                "object_id": object_id,
                "view_idx": view_idx,
                "target_lighting_index": target_lighting_index,
                "cond_lighting_index": cond_lighting_index,
                "image_split": "training",
                "lighting_split": "training",
                "input_mode": input_mode,
                "env_name": env_name,
                "target_file": gt_path.name,
                "input_path": str(input_path),
                "gt_path": str(gt_path),
                "target_lighting_ldr_path": str(target_ldr),
                "target_lighting_hdr_path": str(target_hdr),
            }


def iter_validation_samples(images_root: Path, lighting_root: Path, allowed_views: set[int] | None):
    for split_name in ("seen_lighting", "unseen_lighting"):
        split_images_root = images_root / split_name
        split_lighting_root = lighting_root / split_name
        if not split_images_root.exists() or not split_lighting_root.exists():
            continue

        for object_dir in sorted(p for p in split_images_root.iterdir() if p.is_dir()):
            object_id = object_dir.name
            dataset_name = infer_dataset_name(object_id, "official_orig")
            ldr_dir = split_lighting_root / "LDR" / object_id
            hdr_dir = split_lighting_root / "HDR_rescaled" / object_id
            if not ldr_dir.exists():
                continue
            if not hdr_dir.exists():
                hdr_dir = split_lighting_root / "HDR_normalized" / object_id
            if not hdr_dir.exists():
                continue

            for gt_path in sorted(object_dir.glob("*.png")):
                match = TARGET_IMAGE_PATTERN.match(gt_path.name)
                if match is None:
                    continue
                view_idx = int(match.group(1))
                target_lighting_index = int(match.group(2))
                env_name = match.group(3)
                if allowed_views is not None and view_idx not in allowed_views:
                    continue
                input_path, input_mode, cond_lighting_index = choose_input_path(object_dir, view_idx, target_lighting_index)
                if input_path is None:
                    continue
                target_ldr = ldr_dir / gt_path.name
                target_hdr = hdr_dir / gt_path.name
                if not target_ldr.exists() or not target_hdr.exists():
                    continue
                yield {
                    "preset": f"val_{'seen' if split_name == 'seen_lighting' else 'unseen'}_all",
                    "source_bucket": "validation",
                    "dataset": dataset_name,
                    "object_id": object_id,
                    "view_idx": view_idx,
                    "target_lighting_index": target_lighting_index,
                    "cond_lighting_index": cond_lighting_index,
                    "image_split": split_name,
                    "lighting_split": split_name,
                    "input_mode": input_mode,
                    "env_name": env_name,
                    "target_file": gt_path.name,
                    "input_path": str(input_path),
                    "gt_path": str(gt_path),
                    "target_lighting_ldr_path": str(target_ldr),
                    "target_lighting_hdr_path": str(target_hdr),
                }


def sample_key(sample: dict):
    return (
        f"{sample['preset']}_{sample['object_id']}"
        f"_v{int(sample['view_idx']):03d}"
        f"_t{int(sample['target_lighting_index']):03d}"
    )


def write_shard(output_dir: Path, shard_index: int, samples: list[dict]):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard_index": shard_index,
        "sample_count": len(samples),
        "samples": samples,
    }
    shard_path = output_dir / "shards" / f"manifest_shard_{shard_index:05d}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return shard_path


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_views = set(args.view_indices) if args.view_indices else None
    summary_counts = defaultdict(Counter)
    shard_paths = []
    shard_samples = []
    shard_index = 0
    total_samples = 0

    def flush():
        nonlocal shard_index, shard_samples
        if not shard_samples:
            return
        shard_index += 1
        shard_paths.append(write_shard(output_dir, shard_index, shard_samples))
        shard_samples = []

    iterators = []
    if args.include_training:
        iterators.append(
            iter_training_samples(
                images_root=resolve_repo_path(args.training_images_root),
                lighting_root=resolve_repo_path(args.training_lighting_root),
                allowed_views=allowed_views,
            )
        )
    if args.include_validation:
        iterators.append(
            iter_validation_samples(
                images_root=resolve_repo_path(args.validation_images_root),
                lighting_root=resolve_repo_path(args.validation_lighting_root),
                allowed_views=allowed_views,
            )
        )

    for iterator in iterators:
        for sample in iterator:
            sample["sample_key"] = sample_key(sample)
            shard_samples.append(sample)
            total_samples += 1
            summary_counts["source_bucket"][sample["source_bucket"]] += 1
            summary_counts["preset"][sample["preset"]] += 1
            summary_counts["dataset"][sample["dataset"]] += 1

            if args.limit_samples is not None and total_samples >= max(int(args.limit_samples), 0):
                flush()
                break
            if len(shard_samples) >= max(int(args.shard_size), 1):
                flush()
        if args.limit_samples is not None and total_samples >= max(int(args.limit_samples), 0):
            break

    flush()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "include_training": args.include_training,
        "include_validation": args.include_validation,
        "training_images_root": str(resolve_repo_path(args.training_images_root)),
        "training_lighting_root": str(resolve_repo_path(args.training_lighting_root)),
        "validation_images_root": str(resolve_repo_path(args.validation_images_root)),
        "validation_lighting_root": str(resolve_repo_path(args.validation_lighting_root)),
        "shard_size": args.shard_size,
        "limit_samples": args.limit_samples,
        "view_indices": sorted(allowed_views) if allowed_views is not None else None,
        "total_samples": total_samples,
        "shard_count": len(shard_paths),
        "counts": {name: dict(counter) for name, counter in summary_counts.items()},
        "shards": [str(path) for path in shard_paths],
    }
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"total_samples={total_samples}")
    print(f"shard_count={len(shard_paths)}")


if __name__ == "__main__":
    main()
