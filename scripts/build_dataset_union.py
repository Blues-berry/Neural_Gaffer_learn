import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


READY_ROOT = REPO_ROOT / "logs" / "ready_subdatasets_20260328"
ORIGINAL_ASSETS_ROOT = Path(
    os.environ.get(
        "NEURAL_GAFFER_ORIGINAL_ASSETS_ROOT",
        REPO_ROOT / "external_data" / "neural_gaffer_original",
    )
)


def read_object_list(path: str | None):
    if not path:
        return None
    list_path = Path(path)
    if not list_path.exists():
        raise FileNotFoundError(f"Object list not found: {list_path}")
    if list_path.suffix.lower() == ".json":
        return json.loads(list_path.read_text())
    return [line.strip() for line in list_path.read_text().splitlines() if line.strip()]


def iter_object_ids(img_dir: Path, list_path: str | None):
    object_list = read_object_list(list_path)
    if object_list is not None:
        return object_list
    return sorted([p.name for p in img_dir.iterdir() if p.is_dir()])


def link_dir(src: Path, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src)


def build_union(output_dir: Path, sources: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    lighting_dir = output_dir / "lighting"
    (lighting_dir / "LDR").mkdir(parents=True, exist_ok=True)
    (lighting_dir / "HDR_rescaled").mkdir(parents=True, exist_ok=True)
    (lighting_dir / "HDR_raw").mkdir(parents=True, exist_ok=True)

    union_ids: set[str] = set()
    per_source_counts = []
    duplicates = 0

    for source in sources:
        img_root = Path(source["img_dir"])
        lighting_root = Path(source["lighting_dir"])
        list_path = source.get("list_path")

        object_ids = iter_object_ids(img_root, list_path)
        added = 0
        skipped = 0

        for obj_id in object_ids:
            if obj_id in union_ids:
                duplicates += 1
                skipped += 1
                continue
            src_img = img_root / obj_id
            src_ldr = lighting_root / "LDR" / obj_id
            src_hdr_rescaled = lighting_root / "HDR_rescaled" / obj_id
            if not src_img.exists() or not src_ldr.exists() or not src_hdr_rescaled.exists():
                skipped += 1
                continue
            union_ids.add(obj_id)
            added += 1

            link_dir(src_img, images_dir / obj_id)
            link_dir(src_ldr, lighting_dir / "LDR" / obj_id)
            link_dir(src_hdr_rescaled, lighting_dir / "HDR_rescaled" / obj_id)
            src_hdr_raw = lighting_root / "HDR_raw" / obj_id
            if src_hdr_raw.exists():
                link_dir(src_hdr_raw, lighting_dir / "HDR_raw" / obj_id)

        per_source_counts.append(
            {
                "name": source["name"],
                "listed": len(object_ids),
                "added": added,
                "skipped": skipped,
                "img_dir": str(img_root),
                "lighting_dir": str(lighting_root),
                "list_path": list_path,
            }
        )

    training_list = sorted(union_ids)
    (images_dir / "training_object_list.json").write_text(json.dumps(training_list, indent=2) + "\n")
    (output_dir / "training_object_list.json").write_text(json.dumps(training_list, indent=2) + "\n")

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "total_objects": len(training_list),
        "duplicates_skipped": duplicates,
        "sources": per_source_counts,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_presets():
    presets = {
        "main": {
            "output_dir": REPO_ROOT / "logs" / "dataset_unions" / "main_current_original_official_ecommerce",
            "sources": [
                {
                    "name": "current_original",
                    "img_dir": str(REPO_ROOT / "training_data" / "images" / "training_img_data_subset"),
                    "lighting_dir": str(REPO_ROOT / "training_data" / "lighting" / "training_lighting_data_subset"),
                    "list_path": None,
                },
                {
                    "name": "official_2000",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_official_2000"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_official_2000"
                    ),
                    "list_path": str(READY_ROOT / "official_2000" / "filtered_objects.txt"),
                },
                {
                    "name": "ecommerce",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_ecommerce_subset"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_ecommerce_subset"
                    ),
                    "list_path": str(READY_ROOT / "ecommerce" / "filtered_objects.txt"),
                },
            ],
        },
        "full": {
            "output_dir": REPO_ROOT / "logs" / "dataset_unions" / "full_current_original_official2000_ecommerce1000_3dfuture",
            "sources": [
                {
                    "name": "current_original",
                    "img_dir": str(REPO_ROOT / "training_data" / "images" / "training_img_data_subset"),
                    "lighting_dir": str(REPO_ROOT / "training_data" / "lighting" / "training_lighting_data_subset"),
                    "list_path": None,
                },
                {
                    "name": "official_2000",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_official_2000"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_official_2000"
                    ),
                    "list_path": str(READY_ROOT / "official_2000" / "filtered_objects.txt"),
                },
                {
                    "name": "ecommerce_1000",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_ecommerce_subset"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_ecommerce_subset"
                    ),
                    "list_path": str(READY_ROOT / "ecommerce" / "filtered_objects.txt"),
                },
                {
                    "name": "three_future",
                    "img_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_three_future_standalone"
                    ),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_three_future_standalone"
                    ),
                    "list_path": str(READY_ROOT / "three_future" / "filtered_objects.txt"),
                },
            ],
        },
        "all_available": {
            "output_dir": REPO_ROOT / "logs" / "dataset_unions" / "full_current_original_official2000_ecommerce1000_3dfuture_landscape",
            "sources": [
                {
                    "name": "current_original",
                    "img_dir": str(REPO_ROOT / "training_data" / "images" / "training_img_data_subset"),
                    "lighting_dir": str(REPO_ROOT / "training_data" / "lighting" / "training_lighting_data_subset"),
                    "list_path": None,
                },
                {
                    "name": "official_2000",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_official_2000"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_official_2000"
                    ),
                    "list_path": str(READY_ROOT / "official_2000" / "filtered_objects.txt"),
                },
                {
                    "name": "ecommerce_1000",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_ecommerce_subset"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_ecommerce_subset"
                    ),
                    "list_path": str(READY_ROOT / "ecommerce" / "filtered_objects.txt"),
                },
                {
                    "name": "three_future",
                    "img_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_three_future_standalone"
                    ),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_three_future_standalone"
                    ),
                    "list_path": str(READY_ROOT / "three_future" / "filtered_objects.txt"),
                },
                {
                    "name": "landscape",
                    "img_dir": str(ORIGINAL_ASSETS_ROOT / "training_data/images/training_img_data_landscape_subset"),
                    "lighting_dir": str(
                        ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_landscape_subset"
                    ),
                    "list_path": str(READY_ROOT / "landscape" / "filtered_objects.txt"),
                },
            ],
        },
    }
    return presets


def main():
    parser = argparse.ArgumentParser(description="Build union training datasets by linking multiple sources.")
    parser.add_argument("--preset", type=str, default="main", choices=["main", "full", "all_available"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    presets = build_presets()
    preset = presets[args.preset]
    output_dir = Path(args.output_dir) if args.output_dir else preset["output_dir"]
    meta = build_union(output_dir, preset["sources"])
    print(f"Wrote union dataset: {output_dir}")
    print(f"Total objects: {meta['total_objects']}")


if __name__ == "__main__":
    main()
