import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_IMAGE_PATTERN = re.compile(r"^(\d{3})_(\d{3})_(.+)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check ground-truth / input / lighting integrity across training and validation unions."
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
    parser.add_argument("--view-indices", nargs="*", type=int, default=None)
    parser.add_argument("--resolution-audit-limit", type=int, default=5000)
    parser.add_argument("--output-json", required=True)
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


def safe_image_info(path: Path):
    with Image.open(path) as image:
        return {
            "size_wh": [int(image.size[0]), int(image.size[1])],
            "mode": str(image.mode),
        }


def append_example(example_store: dict[str, list[dict]], key: str, payload: dict, limit: int = 10):
    bucket = example_store.setdefault(key, [])
    if len(bucket) < limit:
        bucket.append(payload)


def inspect_target(
    *,
    sample: dict,
    counters: dict[str, Counter],
    examples: dict[str, list[dict]],
    resolution_state: dict,
):
    counters["global"]["target_files"] += 1
    counters["source_bucket"][sample["source_bucket"]] += 1
    counters["dataset"][sample["dataset"]] += 1
    counters["input_mode"][sample.get("input_mode") or "missing"] += 1

    missing = []
    required_paths = {
        "gt_missing": sample["gt_path"],
        "input_missing": sample["input_path"],
        "ldr_missing": sample["target_lighting_ldr_path"],
        "hdr_missing": sample["target_lighting_hdr_path"],
        "rt_missing": sample["rt_path"],
    }
    optional_paths = {
        "normals_missing": sample["normals_path"],
    }

    for issue_name, path in required_paths.items():
        if not Path(path).exists():
            counters["issues"][issue_name] += 1
            missing.append(issue_name)
            append_example(examples, issue_name, sample)

    for issue_name, path in optional_paths.items():
        if path and not Path(path).exists():
            counters["issues"][issue_name] += 1
            append_example(examples, issue_name, sample)

    if not missing:
        counters["global"]["complete_required_sets"] += 1

    audit_limit = resolution_state["limit"]
    if resolution_state["count"] >= audit_limit:
        return
    if missing:
        return

    gt_info = safe_image_info(Path(sample["gt_path"]))
    input_info = safe_image_info(Path(sample["input_path"]))
    ldr_info = safe_image_info(Path(sample["target_lighting_ldr_path"]))
    hdr_info = safe_image_info(Path(sample["target_lighting_hdr_path"]))

    resolution_state["count"] += 1
    resolution_state["gt_sizes"][tuple(gt_info["size_wh"])] += 1
    resolution_state["input_sizes"][tuple(input_info["size_wh"])] += 1
    resolution_state["ldr_sizes"][tuple(ldr_info["size_wh"])] += 1
    resolution_state["hdr_sizes"][tuple(hdr_info["size_wh"])] += 1
    resolution_state["gt_modes"][gt_info["mode"]] += 1
    resolution_state["input_modes"][input_info["mode"]] += 1

    if input_info["size_wh"] != gt_info["size_wh"]:
        counters["issues"]["input_gt_size_mismatch"] += 1
        append_example(
            examples,
            "input_gt_size_mismatch",
            {
                **sample,
                "gt_size_wh": gt_info["size_wh"],
                "input_size_wh": input_info["size_wh"],
            },
        )
    if ldr_info["size_wh"] != gt_info["size_wh"]:
        counters["issues"]["ldr_gt_size_mismatch"] += 1
        append_example(
            examples,
            "ldr_gt_size_mismatch",
            {
                **sample,
                "gt_size_wh": gt_info["size_wh"],
                "ldr_size_wh": ldr_info["size_wh"],
            },
        )
    if hdr_info["size_wh"] != gt_info["size_wh"]:
        counters["issues"]["hdr_gt_size_mismatch"] += 1
        append_example(
            examples,
            "hdr_gt_size_mismatch",
            {
                **sample,
                "gt_size_wh": gt_info["size_wh"],
                "hdr_size_wh": hdr_info["size_wh"],
            },
        )


def iter_training_samples(images_root: Path, lighting_root: Path, allowed_views: set[int] | None):
    for object_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        object_id = object_dir.name
        dataset_name = infer_dataset_name(object_id, "official_1000_train")
        ldr_dir = lighting_root / "LDR" / object_id
        hdr_dir = lighting_root / "HDR_rescaled" / object_id
        if not hdr_dir.exists():
            hdr_dir = lighting_root / "HDR_normalized" / object_id

        for gt_path in sorted(object_dir.glob("*.png")):
            match = TARGET_IMAGE_PATTERN.match(gt_path.name)
            if match is None:
                continue
            view_idx = int(match.group(1))
            target_lighting_index = int(match.group(2))
            if allowed_views is not None and view_idx not in allowed_views:
                continue
            input_path, input_mode, cond_lighting_index = choose_input_path(object_dir, view_idx, target_lighting_index)
            yield {
                "source_bucket": "training",
                "split": "training",
                "dataset": dataset_name,
                "object_id": object_id,
                "view_idx": view_idx,
                "target_lighting_index": target_lighting_index,
                "cond_lighting_index": cond_lighting_index,
                "input_mode": input_mode,
                "target_file": gt_path.name,
                "input_path": str(input_path) if input_path else "",
                "gt_path": str(gt_path),
                "target_lighting_ldr_path": str(ldr_dir / gt_path.name),
                "target_lighting_hdr_path": str(hdr_dir / gt_path.name),
                "rt_path": str(object_dir / f"{view_idx:03d}_RT.npy"),
                "normals_path": str(object_dir / f"{view_idx:03d}_normals.png"),
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
            if not hdr_dir.exists():
                hdr_dir = split_lighting_root / "HDR_normalized" / object_id
            for gt_path in sorted(object_dir.glob("*.png")):
                match = TARGET_IMAGE_PATTERN.match(gt_path.name)
                if match is None:
                    continue
                view_idx = int(match.group(1))
                target_lighting_index = int(match.group(2))
                if allowed_views is not None and view_idx not in allowed_views:
                    continue
                input_path, input_mode, cond_lighting_index = choose_input_path(object_dir, view_idx, target_lighting_index)
                yield {
                    "source_bucket": "validation",
                    "split": split_name,
                    "dataset": dataset_name,
                    "object_id": object_id,
                    "view_idx": view_idx,
                    "target_lighting_index": target_lighting_index,
                    "cond_lighting_index": cond_lighting_index,
                    "input_mode": input_mode,
                    "target_file": gt_path.name,
                    "input_path": str(input_path) if input_path else "",
                    "gt_path": str(gt_path),
                    "target_lighting_ldr_path": str(ldr_dir / gt_path.name),
                    "target_lighting_hdr_path": str(hdr_dir / gt_path.name),
                    "rt_path": str(object_dir / f"{view_idx:03d}_RT.npy"),
                    "normals_path": str(object_dir / f"{view_idx:03d}_normals.png"),
                }


def summarize_counter(counter: Counter):
    return {str(key): int(value) for key, value in counter.most_common()}


def main():
    args = parse_args()
    allowed_views = set(args.view_indices) if args.view_indices else None
    counters = defaultdict(Counter)
    examples = {}
    resolution_state = {
        "limit": max(int(args.resolution_audit_limit), 0),
        "count": 0,
        "gt_sizes": Counter(),
        "input_sizes": Counter(),
        "ldr_sizes": Counter(),
        "hdr_sizes": Counter(),
        "gt_modes": Counter(),
        "input_modes": Counter(),
    }

    if args.include_training:
        for sample in iter_training_samples(
            resolve_repo_path(args.training_images_root),
            resolve_repo_path(args.training_lighting_root),
            allowed_views,
        ):
            inspect_target(sample=sample, counters=counters, examples=examples, resolution_state=resolution_state)

    if args.include_validation:
        for sample in iter_validation_samples(
            resolve_repo_path(args.validation_images_root),
            resolve_repo_path(args.validation_lighting_root),
            allowed_views,
        ):
            inspect_target(sample=sample, counters=counters, examples=examples, resolution_state=resolution_state)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_training": args.include_training,
        "include_validation": args.include_validation,
        "view_indices": sorted(allowed_views) if allowed_views else None,
        "global_counts": summarize_counter(counters["global"]),
        "issue_counts": summarize_counter(counters["issues"]),
        "source_bucket_counts": summarize_counter(counters["source_bucket"]),
        "dataset_counts": summarize_counter(counters["dataset"]),
        "input_mode_counts": summarize_counter(counters["input_mode"]),
        "resolution_audit": {
            "audited_sample_count": int(resolution_state["count"]),
            "gt_size_counts": summarize_counter(resolution_state["gt_sizes"]),
            "input_size_counts": summarize_counter(resolution_state["input_sizes"]),
            "ldr_size_counts": summarize_counter(resolution_state["ldr_sizes"]),
            "hdr_size_counts": summarize_counter(resolution_state["hdr_sizes"]),
            "gt_mode_counts": summarize_counter(resolution_state["gt_modes"]),
            "input_mode_counts": summarize_counter(resolution_state["input_modes"]),
        },
        "examples": examples,
    }

    output_json = resolve_repo_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
