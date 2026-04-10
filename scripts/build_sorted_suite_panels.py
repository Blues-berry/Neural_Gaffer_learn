import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build sorted comparison panels for an arbitrary suite of methods."
    )
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--per-sample-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--primary-method", required=True)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--preserve-native-size", action="store_true")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--visual-tag", default="input_white_methods_gt_hdrbg")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_key_from_sample(sample: dict):
    return sample.get("sample_key") or (
        f"{sample.get('preset', 'na')}_{sample.get('object_id', 'unknown')}"
        f"_v{int(sample.get('view_idx', 0)):03d}"
        f"_t{int(sample.get('target_lighting_index', 0)):03d}"
    )


def safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def infer_uniform_tile_size(samples: list[dict], ground_truth_image_key: str, fallback: int = 256):
    if not samples:
        return int(fallback)
    candidate_sides = []
    for sample in samples:
        gt_path = (
            sample.get("ground_truth_composited_export")
            if ground_truth_image_key == "composited"
            else sample.get("ground_truth_white_export") or sample.get("ground_truth_export")
        )
        if not gt_path:
            continue
        path = Path(gt_path)
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                candidate_sides.append(int(max(image.size)))
        except Exception:
            continue
    if not candidate_sides:
        return int(fallback)
    counts = defaultdict(int)
    for value in candidate_sides:
        counts[value] += 1
    return int(max(counts.items(), key=lambda item: (item[1], item[0]))[0])


def write_subset_manifest(samples: list[dict], output_path: Path, selection_name: str, visual_mode: str):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_name": selection_name,
        "visual_mode": visual_mode,
        "samples": deepcopy(samples),
    }
    dump_json(output_path, payload)
    return output_path


def chunk_samples(samples: list[dict], page_size: int):
    for index in range(0, len(samples), page_size):
        yield index // page_size + 1, samples[index:index + page_size]


def run_cmd(cmd: list[str]):
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, check=True)


def build_panel_variants(
    manifest_path: Path,
    output_base: Path,
    columns: list[str],
    preserve_native_size: bool,
    tile_size: int,
    method_image_key: str,
    input_image_key: str,
    ground_truth_image_key: str,
):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    headers_path = output_base.with_name(f"{output_base.stem}_headers{output_base.suffix}")
    no_text_path = output_base.with_name(f"{output_base.stem}_no_text{output_base.suffix}")
    legacy_headers_path = output_base

    base_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
        "--assets-manifest",
        str(manifest_path),
        "--columns",
        *columns,
        "--method-image-key",
        method_image_key,
        "--input-image-key",
        input_image_key,
        "--ground-truth-image-key",
        ground_truth_image_key,
        "--padding",
        "14",
        "--header-height",
        "60",
        "--hide-row-labels",
    ]
    if preserve_native_size:
        base_cmd.append("--preserve-native-size")
    else:
        base_cmd.extend(["--tile-size", str(int(tile_size))])

    run_cmd([*base_cmd, "--output", str(headers_path)])
    run_cmd([*base_cmd, "--output", str(no_text_path), "--no-text"])
    if headers_path != legacy_headers_path:
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
                "--assets-manifest",
                str(manifest_path),
                "--output",
                str(legacy_headers_path),
                "--columns",
                *columns,
                "--method-image-key",
                method_image_key,
                "--input-image-key",
                input_image_key,
                "--ground-truth-image-key",
                ground_truth_image_key,
                "--padding",
                "14",
                "--header-height",
                "60",
                "--hide-row-labels",
                *(["--preserve-native-size"] if preserve_native_size else ["--tile-size", str(int(tile_size))]),
            ]
        )
    return {
        "headers": str(headers_path),
        "no_text": str(no_text_path),
        "legacy": str(legacy_headers_path),
    }


def build_all_method_panels(
    samples: list[dict],
    selection_name: str,
    methods: list[str],
    output_root: Path,
    page_size: int,
    preserve_native_size: bool,
    tile_size: int,
    method_image_key: str,
    input_image_key: str,
    ground_truth_image_key: str,
    visual_tag: str,
):
    panel_paths = []
    panel_dir = output_root / "panels" / selection_name / visual_tag
    manifest_dir = output_root / "panel_manifests" / selection_name / visual_tag
    columns = ["input_image"] + [f"method:{name}" for name in methods] + ["ground_truth", "target_lighting"]
    for page_number, page_samples in chunk_samples(samples, page_size):
        manifest_path = manifest_dir / f"all_methods_page_{page_number:02d}.json"
        write_subset_manifest(page_samples, manifest_path, selection_name=selection_name, visual_mode=visual_tag)
        output_path = panel_dir / f"all_methods_page_{page_number:02d}.png"
        panel_paths.append(
            build_panel_variants(
                manifest_path=manifest_path,
                output_base=output_path,
                columns=columns,
                preserve_native_size=preserve_native_size,
                tile_size=tile_size,
                method_image_key=method_image_key,
                input_image_key=input_image_key,
                ground_truth_image_key=ground_truth_image_key,
            )
        )
    return panel_paths


def build_pair_panels(
    samples: list[dict],
    selection_name: str,
    methods: list[str],
    primary_method: str,
    output_root: Path,
    page_size: int,
    preserve_native_size: bool,
    tile_size: int,
    method_image_key: str,
    input_image_key: str,
    ground_truth_image_key: str,
    visual_tag: str,
):
    pair_paths = {}
    for method_name in methods:
        if method_name == primary_method:
            continue
        panel_dir = output_root / "panels" / selection_name / visual_tag
        manifest_dir = output_root / "panel_manifests" / selection_name / visual_tag
        outputs = []
        for page_number, page_samples in chunk_samples(samples, page_size):
            manifest_path = manifest_dir / f"{method_name}_vs_{primary_method}_page_{page_number:02d}.json"
            write_subset_manifest(page_samples, manifest_path, selection_name=selection_name, visual_mode=visual_tag)
            output_path = panel_dir / f"{method_name}_vs_{primary_method}_page_{page_number:02d}.png"
            outputs.append(
                build_panel_variants(
                    manifest_path=manifest_path,
                    output_base=output_path,
                    columns=[
                        "input_image",
                        f"method:{method_name}",
                        f"method:{primary_method}",
                        "ground_truth",
                        "target_lighting",
                    ],
                    preserve_native_size=preserve_native_size,
                    tile_size=tile_size,
                    method_image_key=method_image_key,
                    input_image_key=input_image_key,
                    ground_truth_image_key=ground_truth_image_key,
                )
            )
        pair_paths[method_name] = outputs
    return pair_paths


def main():
    args = parse_args()
    assets_manifest_path = resolve_repo_path(args.assets_manifest)
    per_sample_csv_path = resolve_repo_path(args.per_sample_csv)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    assets_manifest = load_json(assets_manifest_path)
    with per_sample_csv_path.open("r", encoding="utf-8", newline="") as handle:
        proxy_rows = list(csv.DictReader(handle))

    metric_index = defaultdict(dict)
    for row in proxy_rows:
        metric_index[row["sample_key"]][row["method"]] = {
            "fg_rmse": safe_float(row.get("fg_rmse"), float("inf")),
            "foreground_psnr": safe_float(row.get("foreground_psnr"), safe_float(row.get("fg_psnr"), float("-inf"))),
            "full_psnr": safe_float(row.get("full_psnr"), float("-inf")),
            "highlight_rmse": safe_float(row.get("highlight_rmse"), float("inf")),
            "highlight_psnr": safe_float(row.get("highlight_psnr"), float("-inf")),
            "highlight_area_abs_error": safe_float(row.get("highlight_area_abs_error"), float("inf")),
            "highlight_saturated_ratio_abs_error": safe_float(row.get("highlight_saturated_ratio_abs_error"), float("inf")),
        }

    primary_rows = []
    for sample in assets_manifest.get("samples", []):
        key = sample_key_from_sample(sample)
        primary_metrics = metric_index.get(key, {}).get(args.primary_method)
        if primary_metrics is None:
            continue
        primary_rows.append((sample, primary_metrics))

    non_highlight_sorted = [
        sample
        for sample, metrics in sorted(
            primary_rows,
            key=lambda item: (
                item[1]["fg_rmse"],
                -item[1]["foreground_psnr"],
                -item[1]["full_psnr"],
                item[1]["highlight_rmse"],
                sample_key_from_sample(item[0]),
            ),
        )
    ]
    highlight_sorted = [
        sample
        for sample, metrics in sorted(
            primary_rows,
            key=lambda item: (
                item[1]["highlight_rmse"],
                -item[1]["highlight_psnr"],
                item[1]["highlight_area_abs_error"],
                item[1]["highlight_saturated_ratio_abs_error"],
                item[1]["fg_rmse"],
                sample_key_from_sample(item[0]),
            ),
        )
    ]

    tile_size = int(args.tile_size or infer_uniform_tile_size(non_highlight_sorted, args.ground_truth_image_key))
    groups = {
        "all_sorted_non_highlight": non_highlight_sorted,
        "all_sorted_highlight": highlight_sorted,
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets_manifest": str(assets_manifest_path),
        "per_sample_csv": str(per_sample_csv_path),
        "methods": list(args.methods),
        "primary_method": args.primary_method,
        "visual_tag": args.visual_tag,
        "page_size": int(args.page_size),
        "tile_size": tile_size,
        "groups": {},
    }

    for group_name, samples in groups.items():
        all_methods = build_all_method_panels(
            samples=samples,
            selection_name=group_name,
            methods=list(args.methods),
            output_root=output_root,
            page_size=args.page_size,
            preserve_native_size=bool(args.preserve_native_size),
            tile_size=tile_size,
            method_image_key=args.method_image_key,
            input_image_key=args.input_image_key,
            ground_truth_image_key=args.ground_truth_image_key,
            visual_tag=args.visual_tag,
        )
        pair_panels = build_pair_panels(
            samples=samples,
            selection_name=group_name,
            methods=list(args.methods),
            primary_method=args.primary_method,
            output_root=output_root,
            page_size=args.page_size,
            preserve_native_size=bool(args.preserve_native_size),
            tile_size=tile_size,
            method_image_key=args.method_image_key,
            input_image_key=args.input_image_key,
            ground_truth_image_key=args.ground_truth_image_key,
            visual_tag=args.visual_tag,
        )
        summary["groups"][group_name] = {
            "sample_count": len(samples),
            "all_methods": all_methods,
            "pair_panels": pair_panels,
        }

    dump_json(output_root / "grouped_panels_summary.json", summary)
    print(f"wrote {output_root / 'grouped_panels_summary.json'}")


if __name__ == "__main__":
    main()
