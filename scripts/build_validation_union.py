import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


READY_ROOT = REPO_ROOT / "logs" / "ready_subdatasets_20260328"


def read_object_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def safe_link(src: Path, dest: Path):
    if dest.exists() or dest.is_symlink():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src)


def resolve_source(source: dict):
    if "seen_img_dir" not in source and "root" in source:
        source_root = Path(source["root"])
        return {
            "name": source["name"],
            "root": str(source_root),
            "seen_img_dir": source_root / "val" / "images" / "seen_lighting",
            "unseen_img_dir": source_root / "val" / "images" / "unseen_lighting",
            "seen_lighting_dir": source_root / "val" / "lighting" / "seen_lighting",
            "unseen_lighting_dir": source_root / "val" / "lighting" / "unseen_lighting",
            "val_seen_list": source_root / "val" / "images" / "seen_lighting" / "val_seen_object_list.json",
            "val_unseen_list": source_root / "val" / "images" / "unseen_lighting" / "val_unseen_object_list.json",
        }

    return {
        "name": source["name"],
        "root": source.get("root", ""),
        "seen_img_dir": Path(source["seen_img_dir"]),
        "unseen_img_dir": Path(source["unseen_img_dir"]),
        "seen_lighting_dir": Path(source["seen_lighting_dir"]),
        "unseen_lighting_dir": Path(source["unseen_lighting_dir"]),
        "val_seen_list": Path(source["val_seen_list"]),
        "val_unseen_list": Path(source["val_unseen_list"]),
    }


def build_validation_union(output_dir: Path, sources: list[dict]):
    seen_img_dir = output_dir / "images" / "seen_lighting"
    unseen_img_dir = output_dir / "images" / "unseen_lighting"
    seen_lighting_dir = output_dir / "lighting" / "seen_lighting"
    unseen_lighting_dir = output_dir / "lighting" / "unseen_lighting"

    for root in [seen_img_dir, unseen_img_dir]:
        root.mkdir(parents=True, exist_ok=True)
    for root in [seen_lighting_dir, unseen_lighting_dir]:
        (root / "LDR").mkdir(parents=True, exist_ok=True)
        (root / "HDR_rescaled").mkdir(parents=True, exist_ok=True)
        (root / "HDR_raw").mkdir(parents=True, exist_ok=True)

    union_seen_ids: list[str] = []
    union_unseen_ids: list[str] = []
    per_source = []

    for source in sources:
        resolved = resolve_source(source)
        source_name = resolved["name"]
        source_seen_img = resolved["seen_img_dir"]
        source_unseen_img = resolved["unseen_img_dir"]
        source_seen_lighting = resolved["seen_lighting_dir"]
        source_unseen_lighting = resolved["unseen_lighting_dir"]

        source_seen_ids = read_object_list(resolved["val_seen_list"])
        source_unseen_ids = read_object_list(resolved["val_unseen_list"])

        added_seen = 0
        added_unseen = 0

        for obj_id in source_seen_ids:
            union_obj_id = f"{source_name}__{obj_id}"
            safe_link(source_seen_img / obj_id, seen_img_dir / union_obj_id)
            safe_link(source_seen_lighting / "LDR" / obj_id, seen_lighting_dir / "LDR" / union_obj_id)
            safe_link(source_seen_lighting / "HDR_rescaled" / obj_id, seen_lighting_dir / "HDR_rescaled" / union_obj_id)
            src_hdr_raw = source_seen_lighting / "HDR_raw" / obj_id
            if src_hdr_raw.exists():
                safe_link(src_hdr_raw, seen_lighting_dir / "HDR_raw" / union_obj_id)
            union_seen_ids.append(union_obj_id)
            added_seen += 1

        for obj_id in source_unseen_ids:
            union_obj_id = f"{source_name}__{obj_id}"
            safe_link(source_unseen_img / obj_id, unseen_img_dir / union_obj_id)
            safe_link(source_unseen_lighting / "LDR" / obj_id, unseen_lighting_dir / "LDR" / union_obj_id)
            safe_link(source_unseen_lighting / "HDR_rescaled" / obj_id, unseen_lighting_dir / "HDR_rescaled" / union_obj_id)
            src_hdr_raw = source_unseen_lighting / "HDR_raw" / obj_id
            if src_hdr_raw.exists():
                safe_link(src_hdr_raw, unseen_lighting_dir / "HDR_raw" / union_obj_id)
            union_unseen_ids.append(union_obj_id)
            added_unseen += 1

        per_source.append(
            {
                "name": source_name,
                "root": resolved["root"],
                "val_seen_count": added_seen,
                "val_unseen_count": added_unseen,
            }
        )

    for image_split_dir in [seen_img_dir, unseen_img_dir]:
        (image_split_dir / "val_seen_object_list.json").write_text(
            json.dumps(sorted(union_seen_ids), indent=2) + "\n"
        )
        (image_split_dir / "val_unseen_object_list.json").write_text(
            json.dumps(sorted(union_unseen_ids), indent=2) + "\n"
        )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "val_seen_total": len(union_seen_ids),
        "val_unseen_total": len(union_unseen_ids),
        "sources": per_source,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_presets():
    return {
        "full_main": {
            "output_dir": REPO_ROOT / "logs" / "dataset_validation_unions" / "full_main_ready_20260329",
            "sources": [
                {"name": "official_2000", "root": str(READY_ROOT / "official_2000")},
                {"name": "ecommerce", "root": str(READY_ROOT / "ecommerce")},
                {"name": "three_future", "root": str(READY_ROOT / "three_future")},
            ],
        },
        "all_ready": {
            "output_dir": REPO_ROOT / "logs" / "dataset_validation_unions" / "all_ready_20260329",
            "sources": [
                {"name": "official_2000", "root": str(READY_ROOT / "official_2000")},
                {"name": "ecommerce", "root": str(READY_ROOT / "ecommerce")},
                {"name": "three_future", "root": str(READY_ROOT / "three_future")},
                {"name": "landscape", "root": str(READY_ROOT / "landscape")},
            ],
        },
        "all_ready_plus_official": {
            "output_dir": REPO_ROOT / "logs" / "dataset_validation_unions" / "all_ready_plus_official_20260403",
            "sources": [
                {"name": "official_2000", "root": str(READY_ROOT / "official_2000")},
                {"name": "ecommerce", "root": str(READY_ROOT / "ecommerce")},
                {"name": "three_future", "root": str(READY_ROOT / "three_future")},
                {"name": "landscape", "root": str(READY_ROOT / "landscape")},
                {
                    "name": "official_orig",
                    "root": str(REPO_ROOT / "validation_data"),
                    "seen_img_dir": str(REPO_ROOT / "validation_data" / "images" / "val_rendered_images_resized" / "validation" / "seen_lighting"),
                    "unseen_img_dir": str(REPO_ROOT / "validation_data" / "images" / "val_rendered_images_resized" / "validation" / "unseen_lighting"),
                    "seen_lighting_dir": str(REPO_ROOT / "validation_data" / "lighting" / "val_preprocessed_environment_resized" / "seen_lighting"),
                    "unseen_lighting_dir": str(REPO_ROOT / "validation_data" / "lighting" / "val_preprocessed_environment_resized" / "unseen_lighting"),
                    "val_seen_list": str(REPO_ROOT / "validation_data" / "images" / "val_rendered_images_resized" / "validation" / "seen_lighting" / "val_seen_object_list.json"),
                    "val_unseen_list": str(REPO_ROOT / "validation_data" / "images" / "val_rendered_images_resized" / "validation" / "unseen_lighting" / "val_unseen_object_list.json"),
                },
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build a validation union from ready subdataset val views.")
    parser.add_argument("--preset", type=str, default="full_main", choices=["full_main", "all_ready", "all_ready_plus_official"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    preset = build_presets()[args.preset]
    output_dir = Path(args.output_dir) if args.output_dir else preset["output_dir"]
    meta = build_validation_union(output_dir, preset["sources"])
    print(f"Wrote validation union: {output_dir}")
    print(f"val_seen_total={meta['val_seen_total']}")
    print(f"val_unseen_total={meta['val_unseen_total']}")


if __name__ == "__main__":
    main()
