import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_IMAGE_PATTERN = re.compile(r"^(\d{3})_(\d{3})_(.+)\.png$")
ENV_PRIORITY = (
    "studio",
    "interior",
    "city",
    "courtyard",
    "white_studio_01",
    "white_studio_02",
    "white_studio_05",
    "photo_studio_01",
    "photo_studio_loft_hall",
    "studio_small_08",
    "studio_small_09",
    "brown_photostudio_02",
    "kiara_interior",
    "117_hdrmaps_com_free_2K",
    "128_hdrmaps_com_free_2K",
    "012_hdrmaps_com_free_2K",
    "064_hdrmaps_com_free_2K",
    "125_hdrmaps_com_free_2K",
)
ENV_PRIORITY_INDEX = {name: index for index, name in enumerate(ENV_PRIORITY)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a same-batch one-HDRI-per-object manifest for a standalone rendered set, "
            "such as highlight_test_samples."
        )
    )
    parser.add_argument("--object-manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--lighting-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="highlight_test_samples")
    parser.add_argument("--preset", default="highlight_test_samebatch_onehdri")
    parser.add_argument("--source-bucket", default="standalone")
    parser.add_argument("--image-split", default="highlight_test")
    parser.add_argument("--lighting-split", default="highlight_test")
    parser.add_argument("--view-idx", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=12)
    parser.add_argument("--require-size", type=int, default=512)
    parser.add_argument("--limit-objects", type=int, default=None)
    parser.add_argument(
        "--fast-assume-native-size",
        action="store_true",
        help="Skip per-image PIL reads and trust the raw render batch native size policy.",
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_object_entries(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            payload = payload["samples"]
        elif isinstance(payload.get("objects"), list):
            payload = payload["objects"]
        elif isinstance(payload.get("models"), list):
            payload = payload["models"]
        else:
            payload = []
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported object manifest payload type: {type(payload)}")
    entries = []
    for item in payload:
        if isinstance(item, str):
            entries.append({"uid": item})
        elif isinstance(item, dict) and item.get("uid"):
            entries.append(item)
    return entries


def parse_target_file(name: str):
    match = TARGET_IMAGE_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3)


def image_info(path: Path):
    with Image.open(path) as image:
        return {
            "size_wh": [int(image.size[0]), int(image.size[1])],
            "mode": str(image.mode),
        }


def pick_hdr_path(lighting_root: Path, object_uid: str, target_file: str):
    for subdir in ("HDR_rescaled", "HDR_normalized", "HDR"):
        candidate = lighting_root / subdir / object_uid / target_file
        if candidate.exists():
            return candidate
    return None


def sample_key(sample: dict):
    return (
        f"{sample['preset']}_{sample['object_id']}"
        f"_v{int(sample['view_idx']):03d}"
        f"_t{int(sample['target_lighting_index']):03d}"
    )


def write_shard(output_dir: Path, shard_index: int, samples: list[dict]):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard_index": int(shard_index),
        "sample_count": len(samples),
        "samples": samples,
    }
    shard_path = output_dir / "shards" / f"manifest_shard_{shard_index:05d}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return shard_path


def choose_candidate(candidates: list[dict], env_usage: Counter):
    def sort_key(candidate: dict):
        env_name = candidate["env_name"]
        return (
            int(env_usage[env_name]),
            int(ENV_PRIORITY_INDEX.get(env_name, len(ENV_PRIORITY) + 10_000)),
            int(candidate["target_lighting_index"]),
            str(candidate["target_file"]),
        )

    return min(candidates, key=sort_key)


def main():
    args = parse_args()
    object_manifest = resolve_repo_path(args.object_manifest)
    raw_root = resolve_repo_path(args.raw_root)
    lighting_root = resolve_repo_path(args.lighting_root)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_entries = load_object_entries(object_manifest)
    issues = defaultdict(list)
    env_usage = Counter()
    samples = []

    for entry in object_entries:
        object_uid = str(entry["uid"])
        raw_object_dir = raw_root / object_uid
        if not raw_object_dir.exists():
            issues["missing_raw_object_dir"].append({"object_id": object_uid, "path": str(raw_object_dir)})
            continue

        input_path = raw_object_dir / f"random_lighting_{int(args.view_idx):03d}.png"
        rt_path = raw_object_dir / f"{int(args.view_idx):03d}_RT.npy"
        if not input_path.exists():
            issues["missing_random_input"].append({"object_id": object_uid, "path": str(input_path)})
            continue
        if not rt_path.exists():
            issues["missing_rt"].append({"object_id": object_uid, "path": str(rt_path)})
            continue

        candidate_rows = []
        for gt_path in sorted(raw_object_dir.glob(f"{int(args.view_idx):03d}_*.png")):
            parsed = parse_target_file(gt_path.name)
            if parsed is None:
                continue
            view_idx, target_lighting_index, env_name = parsed
            ldr_path = lighting_root / "LDR" / object_uid / gt_path.name
            hdr_path = pick_hdr_path(lighting_root, object_uid, gt_path.name)
            if not ldr_path.exists() or hdr_path is None:
                continue
            candidate_rows.append(
                {
                    "view_idx": view_idx,
                    "target_lighting_index": target_lighting_index,
                    "env_name": env_name,
                    "target_file": gt_path.name,
                    "gt_path": gt_path,
                    "ldr_path": ldr_path,
                    "hdr_path": hdr_path,
                }
            )

        if not candidate_rows:
            issues["no_valid_targets"].append({"object_id": object_uid, "raw_object_dir": str(raw_object_dir)})
            continue

        chosen = choose_candidate(candidate_rows, env_usage)
        if args.fast_assume_native_size:
            assumed_size = [int(args.require_size), int(args.require_size)] if int(args.require_size) > 0 else None
            input_info = {"size_wh": assumed_size, "mode": "assumed_from_raw_batch_policy"}
            gt_info = {"size_wh": assumed_size, "mode": "assumed_from_raw_batch_policy"}
        else:
            input_info = image_info(input_path)
            gt_info = image_info(chosen["gt_path"])

        if input_info["size_wh"] != gt_info["size_wh"]:
            issues["input_gt_size_mismatch"].append(
                {
                    "object_id": object_uid,
                    "input_size_wh": input_info["size_wh"],
                    "gt_size_wh": gt_info["size_wh"],
                }
            )
            continue

        required_size = max(int(args.require_size), 0)
        if required_size > 0 and input_info["size_wh"] != [required_size, required_size]:
            issues["unexpected_native_size"].append(
                {"object_id": object_uid, "size_wh": input_info["size_wh"], "required_size": required_size}
            )
            continue

        env_usage[chosen["env_name"]] += 1
        object_id = f"{args.dataset_name}__{object_uid}"
        sample = {
            "preset": str(args.preset),
            "source_bucket": str(args.source_bucket),
            "dataset": str(args.dataset_name),
            "object_id": object_id,
            "plain_object_id": object_uid,
            "view_idx": int(chosen["view_idx"]),
            "target_lighting_index": int(chosen["target_lighting_index"]),
            "cond_lighting_index": None,
            "image_split": str(args.image_split),
            "lighting_split": str(args.lighting_split),
            "input_mode": "random_lighting",
            "env_name": str(chosen["env_name"]),
            "target_file": str(chosen["target_file"]),
            "input_path": str(input_path),
            "gt_path": str(chosen["gt_path"]),
            "target_lighting_ldr_path": str(chosen["ldr_path"]),
            "target_lighting_hdr_path": str(chosen["hdr_path"]),
            "same_render_batch_verified": True,
            "same_render_batch_source_kind": "standalone_render_raw",
            "same_render_batch_resolution_policy": "native_512_same_batch",
            "recommended_loader_resolution": 256,
            "render_batch_root": str(raw_object_dir),
            "input_source_path": str(input_path),
            "gt_source_path": str(chosen["gt_path"]),
            "input_source_size_wh": input_info["size_wh"],
            "gt_source_size_wh": gt_info["size_wh"],
            "input_source_mode": input_info["mode"],
            "gt_source_mode": gt_info["mode"],
            "size_verification_mode": "assumed" if args.fast_assume_native_size else "opened",
            "material_family": entry.get("material_family"),
            "highlight_reason": entry.get("highlight_reason"),
            "local_model_path": entry.get("path"),
        }
        sample["sample_key"] = sample_key(sample)
        samples.append(sample)

        if args.limit_objects is not None and len(samples) >= max(int(args.limit_objects), 0):
            break

    shard_paths = []
    shard_size = max(int(args.shard_size), 1)
    for shard_index, start in enumerate(range(0, len(samples), shard_size), start=1):
        shard_paths.append(write_shard(output_dir, shard_index, samples[start:start + shard_size]))

    full_manifest_path = output_dir / "same_batch_manifest_native512.json"
    full_manifest_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "object_manifest": str(object_manifest),
                "raw_root": str(raw_root),
                "lighting_root": str(lighting_root),
                "dataset": str(args.dataset_name),
                "sample_count": len(samples),
                "samples": samples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    audit_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_manifest": str(object_manifest),
        "raw_root": str(raw_root),
        "lighting_root": str(lighting_root),
        "dataset": str(args.dataset_name),
        "sample_count": len(samples),
        "env_usage": dict(env_usage),
        "issue_counts": {key: len(value) for key, value in issues.items()},
        "issue_examples": {key: value[:20] for key, value in issues.items()},
        "full_manifest": str(full_manifest_path),
        "shards": [str(path) for path in shard_paths],
    }
    audit_path = output_dir / "same_batch_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "object_manifest": str(object_manifest),
        "raw_root": str(raw_root),
        "lighting_root": str(lighting_root),
        "dataset": str(args.dataset_name),
        "sample_count": len(samples),
        "env_usage": dict(env_usage),
        "shard_count": len(shard_paths),
        "shard_size": shard_size,
        "require_size": int(args.require_size),
        "limit_objects": args.limit_objects,
        "fast_assume_native_size": bool(args.fast_assume_native_size),
        "same_batch_policy": {
            "input_gt_must_share_raw_object_dir": True,
            "native_source_resolution_required": [int(args.require_size), int(args.require_size)] if int(args.require_size) > 0 else None,
            "recommended_loader_resolution": 256,
            "one_hdri_per_object": True,
        },
        "full_manifest": str(full_manifest_path),
        "audit_path": str(audit_path),
        "shards": [str(path) for path in shard_paths],
    }
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {summary_path}")
    print(f"sample_count={len(samples)}")
    print(f"shard_count={len(shard_paths)}")


if __name__ == "__main__":
    main()
