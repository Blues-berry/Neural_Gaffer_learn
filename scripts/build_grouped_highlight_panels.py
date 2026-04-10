import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITOR_METHODS = ("baseline", "dilightnet", "rgbx")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build grouped white-background relighting panels for OURS-best, non-OURS-best, "
            "minimal-highlight-difference, and globally sorted metric-based subsets."
        )
    )
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--proxy-per-sample-csv", required=True)
    parser.add_argument("--proxy-summary-json", default=None)
    parser.add_argument("--best-ours-method", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--preserve-native-size", action="store_true")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--visual-tag", default=None)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of groups to render. Defaults to all built-in groups.",
    )
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


def load_proxy_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def resolve_visual_tag(args):
    if args.visual_tag:
        return str(args.visual_tag)
    if (
        args.input_image_key == "white"
        and args.method_image_key == "composited"
        and args.ground_truth_image_key == "composited"
    ):
        return "input_white_scene_bg"
    return (
        f"input_{args.input_image_key}"
        f"__methods_{args.method_image_key}"
        f"__gt_{args.ground_truth_image_key}"
    )


def write_subset_manifest(samples: list[dict], output_path: Path, selection_name: str, visual_mode: str):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_name": selection_name,
        "visual_mode": visual_mode,
        "samples": deepcopy(samples),
    }
    dump_json(output_path, payload)
    return output_path


def infer_uniform_tile_size(samples: list[dict], ground_truth_image_key: str, fallback: int = 256):
    if not samples:
        return int(fallback)

    candidate_sides = []
    for sample in samples:
        if ground_truth_image_key == "composited":
            gt_path = sample.get("ground_truth_composited_export")
        else:
            gt_path = sample.get("ground_truth_white_export") or sample.get("ground_truth_export")
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
    return int(Counter(candidate_sides).most_common(1)[0][0])


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
    best_ours_method: str,
    output_root: Path,
    page_size: int,
    preserve_native_size: bool,
    tile_size: int,
    method_image_key: str,
    input_image_key: str,
    ground_truth_image_key: str,
    visual_tag: str,
):
    if not samples:
        return []
    panel_paths = []
    panel_dir = output_root / "panels" / selection_name / visual_tag
    manifest_dir = output_root / "panel_manifests" / selection_name / visual_tag
    for page_number, page_samples in chunk_samples(samples, page_size):
        manifest_path = manifest_dir / f"all_methods_page_{page_number:02d}.json"
        write_subset_manifest(page_samples, manifest_path, selection_name=selection_name, visual_mode=visual_tag)
        output_path = panel_dir / f"all_methods_page_{page_number:02d}.png"
        panel_paths.append(
            build_panel_variants(
                manifest_path=manifest_path,
                output_base=output_path,
                columns=[
                    "input_image",
                    "method:baseline",
                    "method:dilightnet",
                    "method:rgbx",
                    f"method:{best_ours_method}",
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
    return panel_paths


def build_pair_panels(
    samples: list[dict],
    selection_name: str,
    competitor: str,
    best_ours_method: str,
    output_root: Path,
    page_size: int,
    preserve_native_size: bool,
    tile_size: int,
    method_image_key: str,
    input_image_key: str,
    ground_truth_image_key: str,
    visual_tag: str,
):
    if not samples:
        return []
    panel_paths = []
    panel_dir = output_root / "panels" / selection_name / visual_tag
    manifest_dir = output_root / "panel_manifests" / selection_name / visual_tag
    for page_number, page_samples in chunk_samples(samples, page_size):
        manifest_path = manifest_dir / f"{competitor}_vs_{best_ours_method}_page_{page_number:02d}.json"
        write_subset_manifest(page_samples, manifest_path, selection_name=selection_name, visual_mode=visual_tag)
        output_path = panel_dir / f"{competitor}_vs_{best_ours_method}_page_{page_number:02d}.png"
        panel_paths.append(
            build_panel_variants(
                manifest_path=manifest_path,
                output_base=output_path,
                columns=[
                    "input_image",
                    f"method:{competitor}",
                    f"method:{best_ours_method}",
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
    return panel_paths


def main():
    args = parse_args()
    assets_manifest_path = resolve_repo_path(args.assets_manifest)
    proxy_csv_path = resolve_repo_path(args.proxy_per_sample_csv)
    proxy_summary_path = resolve_repo_path(args.proxy_summary_json)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_tag = resolve_visual_tag(args)

    assets_manifest = load_json(assets_manifest_path)
    proxy_rows = load_proxy_rows(proxy_csv_path)
    uniform_tile_size = int(args.tile_size or infer_uniform_tile_size(assets_manifest.get("samples", []), args.ground_truth_image_key))
    if args.best_ours_method:
        best_ours_method = args.best_ours_method
    elif proxy_summary_path and proxy_summary_path.exists():
        best_ours_method = load_json(proxy_summary_path)["best_ours_method"]
    else:
        raise ValueError("best OURS method is required when --proxy-summary-json is not provided.")

    metric_index = defaultdict(dict)
    for row in proxy_rows:
        metric_index[row["sample_key"]][row["method"]] = {
            "fg_rmse": safe_float(row.get("fg_rmse"), float("inf")),
            "fg_psnr": safe_float(row.get("fg_psnr"), float("-inf")),
            "full_psnr": safe_float(row.get("full_psnr"), float("-inf")),
            "foreground_psnr": safe_float(row.get("foreground_psnr"), float("-inf")),
            "highlight_rmse": safe_float(row.get("highlight_rmse"), float("inf")),
            "highlight_psnr": safe_float(row.get("highlight_psnr"), float("-inf")),
            "highlight_area_abs_error": safe_float(row.get("highlight_area_abs_error"), float("inf")),
            "highlight_saturated_ratio_abs_error": safe_float(row.get("highlight_saturated_ratio_abs_error"), float("inf")),
        }

    ours_best_samples = []
    not_ours_best_samples = []
    ours_highlight_ranked = []
    ours_non_highlight_ranked = []
    sample_outcomes = []

    for sample in assets_manifest.get("samples", []):
        key = sample_key_from_sample(sample)
        methods = {
            method_name: metric_index.get(key, {}).get(method_name)
            for method_name in COMPETITOR_METHODS + (best_ours_method,)
        }
        methods = {name: metrics for name, metrics in methods.items() if metrics is not None}
        if not methods or best_ours_method not in methods:
            continue

        winner = min(
            methods.items(),
            key=lambda item: (
                float(item[1]["fg_rmse"]),
                float(item[1]["highlight_rmse"]),
                -float(item[1]["fg_psnr"]),
                item[0],
            ),
        )[0]
        non_highlight_rank = (
            float(methods[best_ours_method]["fg_rmse"]),
            -float(methods[best_ours_method].get("foreground_psnr", methods[best_ours_method]["fg_psnr"])),
            -float(methods[best_ours_method].get("full_psnr", float("-inf"))),
            key,
        )
        highlight_rank = (
            float(methods[best_ours_method]["highlight_rmse"]),
            -float(methods[best_ours_method].get("highlight_psnr", float("-inf"))),
            float(methods[best_ours_method].get("highlight_area_abs_error", float("inf"))),
            float(methods[best_ours_method].get("highlight_saturated_ratio_abs_error", float("inf"))),
            key,
        )
        record = {
            "sample_key": key,
            "winner": winner,
            "best_ours_method": best_ours_method,
            "method_metrics": methods,
        }
        sample_outcomes.append(record)
        record["best_ours_non_highlight_rank"] = list(non_highlight_rank[:-1])
        record["best_ours_highlight_rank"] = list(highlight_rank[:-1])
        if "full_psnr" in methods[best_ours_method]:
            record["best_ours_full_psnr"] = methods[best_ours_method]["full_psnr"]
        if "highlight_psnr" in methods[best_ours_method]:
            record["best_ours_highlight_psnr"] = methods[best_ours_method]["highlight_psnr"]
        ours_highlight_ranked.append((highlight_rank, sample))
        # Keep a separate ordering over all samples using non-highlight fidelity.
        ours_non_highlight_ranked.append((non_highlight_rank, sample))
        if winner == best_ours_method:
            ours_best_samples.append(sample)
        else:
            not_ours_best_samples.append(sample)

    ours_highlight_ranked.sort(key=lambda item: item[0])
    ours_non_highlight_ranked.sort(key=lambda item: item[0])
    ours_highlight_min_samples = [sample for _, sample in ours_highlight_ranked]
    all_sorted_non_highlight_samples = [sample for _, sample in ours_non_highlight_ranked]
    all_sorted_highlight_samples = [sample for _, sample in ours_highlight_ranked]

    groups = {
        "ours_best": ours_best_samples,
        "not_ours_best": not_ours_best_samples,
        "ours_min_highlight_diff": ours_highlight_min_samples,
        "all_sorted_non_highlight": all_sorted_non_highlight_samples,
        "all_sorted_highlight": all_sorted_highlight_samples,
    }
    if args.groups:
        requested_groups = [name for name in args.groups if name in groups]
        groups = {name: groups[name] for name in requested_groups}

    panels_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_ours_method": best_ours_method,
        "page_size": args.page_size,
        "visual_tag": visual_tag,
        "preserve_native_size": bool(args.preserve_native_size),
        "tile_size": int(uniform_tile_size),
        "method_image_key": args.method_image_key,
        "input_image_key": args.input_image_key,
        "ground_truth_image_key": args.ground_truth_image_key,
        "groups": {},
    }
    for group_name, group_samples in groups.items():
        panels_summary["groups"][group_name] = {
            "sample_count": len(group_samples),
            "all_methods": build_all_method_panels(
                group_samples,
                selection_name=group_name,
                best_ours_method=best_ours_method,
                output_root=output_root,
                page_size=args.page_size,
                preserve_native_size=args.preserve_native_size,
                tile_size=uniform_tile_size,
                method_image_key=args.method_image_key,
                input_image_key=args.input_image_key,
                ground_truth_image_key=args.ground_truth_image_key,
                visual_tag=visual_tag,
            ),
            "pairs": {
                competitor: build_pair_panels(
                    group_samples,
                    selection_name=group_name,
                    competitor=competitor,
                    best_ours_method=best_ours_method,
                    output_root=output_root,
                    page_size=args.page_size,
                    preserve_native_size=args.preserve_native_size,
                    tile_size=uniform_tile_size,
                    method_image_key=args.method_image_key,
                    input_image_key=args.input_image_key,
                    ground_truth_image_key=args.ground_truth_image_key,
                    visual_tag=visual_tag,
                )
                for competitor in COMPETITOR_METHODS
            },
        }

    dump_json(output_root / "grouped_panels_summary.json", panels_summary)
    dump_json(
        output_root / "grouped_sample_outcomes.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_ours_method": best_ours_method,
            "sample_count": len(sample_outcomes),
            "samples": sample_outcomes,
        },
    )
    print(f"wrote {output_root / 'grouped_panels_summary.json'}")


if __name__ == "__main__":
    main()
