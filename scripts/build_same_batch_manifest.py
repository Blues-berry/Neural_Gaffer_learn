import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ASSETS_ROOT = Path(
    os.environ.get(
        "NEURAL_GAFFER_ORIGINAL_ASSETS_ROOT",
        REPO_ROOT / "external_data" / "neural_gaffer_original",
    )
)

DEFAULT_RAW_ROOTS = {
    "official_2000": ORIGINAL_ASSETS_ROOT / "objaverse_jobs/official_2000/raw",
    "ecommerce": ORIGINAL_ASSETS_ROOT / "objaverse_jobs/ecommerce/raw",
    "landscape": ORIGINAL_ASSETS_ROOT / "objaverse_jobs/landscape/raw",
}

DEFAULT_LIGHTING_ROOTS = {
    "official_2000": ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_official_2000",
    "ecommerce": ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_ecommerce_subset",
    "landscape": ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_landscape_subset",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite a comparison manifest so input and GT come from the same raw Objaverse render batch."
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest-name", default="same_batch_manifest_native512.json")
    parser.add_argument("--copy-files", action="store_true", help="Copy staged files instead of creating symlinks.")
    parser.add_argument(
        "--raw-root",
        action="append",
        default=[],
        help="Override raw root mapping with dataset=/abs/path. May be passed multiple times.",
    )
    parser.add_argument(
        "--lighting-root",
        action="append",
        default=[],
        help="Override lighting root mapping with dataset=/abs/path. May be passed multiple times.",
    )
    return parser.parse_args()


def parse_mapping(items, default_map):
    resolved = {key: Path(value) for key, value in default_map.items()}
    for item in items:
        key, value = item.split("=", 1)
        resolved[key.strip()] = Path(value).expanduser()
    return resolved


def derive_dataset_and_plain_id(sample: dict):
    object_id = str(sample["object_id"])
    dataset = sample.get("dataset")
    if "__" in object_id:
        prefix, plain_id = object_id.split("__", 1)
        return str(dataset or prefix), plain_id
    return str(dataset or "unknown"), object_id


def image_info(path: Path):
    with Image.open(path) as image:
        return {
            "size_wh": [int(image.size[0]), int(image.size[1])],
            "mode": str(image.mode),
        }


def ensure_link_or_copy(src: Path, dst: Path, copy_files: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def pick_hdr_path(lighting_root: Path, plain_object_id: str, target_file: str):
    candidates = [
        lighting_root / "HDR_rescaled" / plain_object_id / target_file,
        lighting_root / "HDR_normalized" / plain_object_id / target_file,
        lighting_root / "HDR" / plain_object_id / target_file,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_roots = parse_mapping(args.raw_root, DEFAULT_RAW_ROOTS)
    lighting_roots = parse_mapping(args.lighting_root, DEFAULT_LIGHTING_ROOTS)

    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    samples = payload["samples"]

    staged_images_root = output_root / "staged" / "images"
    staged_lighting_root = output_root / "staged" / "lighting"
    rewritten_samples = []
    issues = defaultdict(list)
    dataset_counts = Counter()

    for sample in samples:
        sample_copy = dict(sample)
        dataset_key, plain_object_id = derive_dataset_and_plain_id(sample_copy)
        union_object_id = str(sample_copy["object_id"])
        dataset_counts[dataset_key] += 1

        raw_root = raw_roots.get(dataset_key)
        lighting_root = lighting_roots.get(dataset_key)
        if raw_root is None:
            issues["missing_raw_root"].append({"object_id": union_object_id, "dataset": dataset_key})
            raise FileNotFoundError(f"No raw root configured for dataset {dataset_key}")
        if lighting_root is None:
            issues["missing_lighting_root"].append({"object_id": union_object_id, "dataset": dataset_key})
            raise FileNotFoundError(f"No lighting root configured for dataset {dataset_key}")

        view_idx = int(sample_copy["view_idx"])
        target_file = str(sample_copy["target_file"])
        image_split = str(sample_copy.get("image_split") or "unseen_lighting")
        lighting_split = str(sample_copy.get("lighting_split") or image_split)

        raw_object_dir = raw_root / plain_object_id
        raw_input = raw_object_dir / f"random_lighting_{view_idx:03d}.png"
        raw_gt = raw_object_dir / target_file
        raw_rt = raw_object_dir / f"{view_idx:03d}_RT.npy"
        raw_normal = raw_object_dir / f"normal_{view_idx:03d}_0001.png"
        ldr_src = lighting_root / "LDR" / plain_object_id / target_file
        hdr_src = pick_hdr_path(lighting_root, plain_object_id, target_file)

        required_paths = {
            "raw_input_missing": raw_input,
            "raw_gt_missing": raw_gt,
            "raw_rt_missing": raw_rt,
            "lighting_ldr_missing": ldr_src,
        }
        for issue_name, path in required_paths.items():
            if not path.exists():
                issues[issue_name].append({"object_id": union_object_id, "path": str(path)})
                raise FileNotFoundError(f"{issue_name}: {path}")
        if hdr_src is None:
            issues["lighting_hdr_missing"].append({"object_id": union_object_id, "target_file": target_file})
            raise FileNotFoundError(f"Missing HDR lighting for {union_object_id} / {target_file}")

        input_info = image_info(raw_input)
        gt_info = image_info(raw_gt)
        same_size = input_info["size_wh"] == gt_info["size_wh"]
        same_mode = input_info["mode"] == gt_info["mode"]
        if not same_size:
            issues["input_gt_size_mismatch"].append(
                {
                    "object_id": union_object_id,
                    "input_size_wh": input_info["size_wh"],
                    "gt_size_wh": gt_info["size_wh"],
                }
            )
            raise RuntimeError(f"Input/GT size mismatch for {union_object_id}: {input_info['size_wh']} vs {gt_info['size_wh']}")

        staged_object_dir = staged_images_root / image_split / union_object_id
        staged_ldr_dir = staged_lighting_root / lighting_split / "LDR" / union_object_id
        staged_hdr_dir = staged_lighting_root / lighting_split / "HDR_rescaled" / union_object_id

        staged_input = staged_object_dir / raw_input.name
        staged_gt = staged_object_dir / raw_gt.name
        staged_rt = staged_object_dir / raw_rt.name
        staged_normal_raw = staged_object_dir / raw_normal.name
        staged_normal_alias = staged_object_dir / f"{view_idx:03d}_normals.png"
        staged_ldr = staged_ldr_dir / target_file
        staged_hdr = staged_hdr_dir / target_file

        ensure_link_or_copy(raw_input, staged_input, args.copy_files)
        ensure_link_or_copy(raw_gt, staged_gt, args.copy_files)
        ensure_link_or_copy(raw_rt, staged_rt, args.copy_files)
        if raw_normal.exists():
            ensure_link_or_copy(raw_normal, staged_normal_raw, args.copy_files)
            if not (staged_normal_alias.exists() or staged_normal_alias.is_symlink()):
                if args.copy_files:
                    import shutil

                    shutil.copy2(raw_normal, staged_normal_alias)
                else:
                    staged_normal_alias.parent.mkdir(parents=True, exist_ok=True)
                    staged_normal_alias.symlink_to(raw_normal)
        ensure_link_or_copy(ldr_src, staged_ldr, args.copy_files)
        ensure_link_or_copy(hdr_src, staged_hdr, args.copy_files)

        sample_copy["input_path"] = str(staged_input)
        sample_copy["gt_path"] = str(staged_gt)
        sample_copy["target_lighting_ldr_path"] = str(staged_ldr)
        sample_copy["target_lighting_hdr_path"] = str(staged_hdr)
        sample_copy["object_id"] = union_object_id
        sample_copy["dataset"] = dataset_key
        sample_copy["same_render_batch_verified"] = True
        sample_copy["render_batch_root"] = str(raw_object_dir)
        sample_copy["input_source_path"] = str(raw_input)
        sample_copy["gt_source_path"] = str(raw_gt)
        sample_copy["input_source_size_wh"] = input_info["size_wh"]
        sample_copy["gt_source_size_wh"] = gt_info["size_wh"]
        sample_copy["input_source_mode"] = input_info["mode"]
        sample_copy["gt_source_mode"] = gt_info["mode"]
        sample_copy["same_render_batch_source_kind"] = "objaverse_jobs_raw"
        sample_copy["same_render_batch_resolution_policy"] = "native_512_same_batch"
        sample_copy["recommended_loader_resolution"] = 256
        sample_copy["same_render_batch_mode_match"] = same_mode
        sample_copy["staged_object_dir"] = str(staged_object_dir)
        rewritten_samples.append(sample_copy)

    out_manifest = dict(payload)
    out_manifest["samples"] = rewritten_samples
    out_manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out_manifest["source_manifest"] = str(source_manifest)
    out_manifest["output_root"] = str(output_root)
    out_manifest["same_render_batch_policy"] = {
        "input_gt_must_share_raw_object_dir": True,
        "native_source_resolution_required": [512, 512],
        "recommended_loader_resolution": 256,
    }

    out_manifest_path = output_root / args.output_manifest_name
    out_manifest_path.write_text(json.dumps(out_manifest, indent=2) + "\n", encoding="utf-8")

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "output_manifest": str(out_manifest_path),
        "output_root": str(output_root),
        "sample_count": len(rewritten_samples),
        "dataset_counts": dict(dataset_counts),
        "issue_counts": {key: len(value) for key, value in issues.items()},
        "issues": issues,
        "raw_roots": {key: str(value) for key, value in raw_roots.items()},
        "lighting_roots": {key: str(value) for key, value in lighting_roots.items()},
    }
    (output_root / "same_batch_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    loader_policy = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(out_manifest_path),
        "source_images_are_native_same_batch_rgba_512": True,
        "recommended_inference_flags": [
            "--no-keep-input-resolution",
            "--resolution",
            "256",
            "--max-resolution",
            "256",
        ],
        "note": "Use native 512x512 raw input/GT provenance, but downsample uniformly to 256x256 at load if you need training/eval parity with Neural Gaffer.",
    }
    (output_root / "loader_policy_256.json").write_text(json.dumps(loader_policy, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out_manifest_path}")
    print(f"wrote {output_root / 'same_batch_audit.json'}")
    print(f"wrote {output_root / 'loader_policy_256.json'}")


if __name__ == "__main__":
    main()
