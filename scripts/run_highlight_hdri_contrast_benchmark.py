import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_ORDER = ["official_2000", "ecommerce", "three_future", "landscape"]
DATASET_LABELS = {
    "official_2000": "office",
    "ecommerce": "ecommerce",
    "three_future": "3d_furniture",
    "landscape": "natural_landscape",
}
OURS_METHODS = ("ours", "ours_full")
CONTRAST_METHODS = ("baseline", "dilightnet", "rgbx")
THEME_ENV_PRIORITY = [
    "HDR_040_Field",
    "087_hdrmaps_com_free_2K",
    "109_hdrmaps_com_free_2K",
    "045_hdrmaps_com_free_2K",
    "117_hdrmaps_com_free_2K",
    "128_hdrmaps_com_free_2K",
    "012_hdrmaps_com_free_2K",
    "064_hdrmaps_com_free_2K",
    "125_hdrmaps_com_free_2K",
    "studio",
    "courtyard",
    "night",
    "city",
    "interior",
    "sunset",
    "forest",
    "photo_studio_01",
    "studio_small_08",
    "studio_small_09",
]
THEME_ENV_SCORE = {name: float(len(THEME_ENV_PRIORITY) - idx) for idx, name in enumerate(THEME_ENV_PRIORITY)}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 500+ combination highlight HDRI contrast benchmark under effects/contrast and effects/best.")
    parser.add_argument("--source-root", default="logs/dataset_validation_unions/all_ready_20260329")
    parser.add_argument("--split", default="unseen_lighting")
    parser.add_argument("--total-groups", type=int, default=512)
    parser.add_argument("--best-per-dataset", type=int, default=8)
    parser.add_argument("--contrast-root", default="effects/contrast/highlight_hdri_contrast_v1")
    parser.add_argument("--best-root", default="effects/best/highlight_hdri_contrast_v1")
    parser.add_argument("--localize-root", default="effects/tmp_local/highlight_hdri_contrast_v1")
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_target_file(target_file: str):
    stem = Path(target_file).stem
    view_token, lighting_token, env_name = stem.split("_", 2)
    return int(view_token), int(lighting_token), env_name


def run_cmd(cmd: list[str], cwd: Path | None = None):
    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=cwd or REPO_ROOT, check=True)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scan_candidates(args):
    source_root = resolve_repo_path(args.source_root)
    images_root = source_root / "images" / args.split
    lighting_root = source_root / "lighting" / args.split

    candidates = []
    env_counter = Counter()
    per_dataset = Counter()
    for object_dir in sorted(images_root.iterdir()):
        if not object_dir.is_dir():
            continue
        object_id = object_dir.name
        dataset = object_id.split("__", 1)[0] if "__" in object_id else "unknown"
        input_path = object_dir / "random_lighting_000.png"
        if not input_path.exists():
            continue

        for gt_path in sorted(object_dir.glob("000_*.png")):
            if gt_path.name == "000_normals.png":
                continue
            view_idx, target_idx, env_name = parse_target_file(gt_path.name)
            if env_name not in THEME_ENV_SCORE:
                continue
            target_ldr_path = lighting_root / "LDR" / object_id / gt_path.name
            target_hdr_path = lighting_root / "HDR_rescaled" / object_id / gt_path.name
            if not target_hdr_path.exists():
                target_hdr_path = lighting_root / "HDR_normalized" / object_id / gt_path.name
            if not target_ldr_path.exists() or not target_hdr_path.exists():
                continue

            selection_score = THEME_ENV_SCORE[env_name]

            candidates.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "object_id": object_id,
                    "view_idx": view_idx,
                    "target_lighting_index": target_idx,
                    "target_file": gt_path.name,
                    "env_name": env_name,
                    "preset": "highlight_contrast",
                    "input_mode": "random_lighting",
                    "input_path": str(input_path),
                    "gt_path": str(gt_path),
                    "target_lighting_ldr_path": str(target_ldr_path),
                    "target_lighting_hdr_path": str(target_hdr_path),
                    "cond_lighting_index": 0,
                    "image_split": args.split,
                    "lighting_split": args.split,
                    "mask_source": "preserved_from_target",
                    "selection_score": float(selection_score),
                    "theme_score": float(selection_score),
                }
            )
            env_counter[env_name] += 1
            per_dataset[dataset] += 1

    return candidates, env_counter, per_dataset


def allocate_balanced_counts(total_groups: int, available_counts: dict[str, int]):
    datasets = [dataset for dataset in DATASET_ORDER if available_counts.get(dataset, 0) > 0]
    if not datasets:
        raise RuntimeError("No available datasets for benchmark selection.")

    base = total_groups // len(datasets)
    remainder = total_groups % len(datasets)
    quotas = {}
    for idx, dataset in enumerate(datasets):
        quotas[dataset] = base + (1 if idx < remainder else 0)

    deficit = 0
    for dataset in datasets:
        if quotas[dataset] > available_counts[dataset]:
            deficit += quotas[dataset] - available_counts[dataset]
            quotas[dataset] = available_counts[dataset]

    while deficit > 0:
        updated = False
        for dataset in datasets:
            if quotas[dataset] < available_counts[dataset]:
                quotas[dataset] += 1
                deficit -= 1
                updated = True
                if deficit == 0:
                    break
        if not updated:
            break
    return quotas


def select_top_candidates(candidates: list[dict], total_groups: int):
    by_dataset_env = defaultdict(lambda: defaultdict(list))
    available_counts = Counter()
    for candidate in candidates:
        by_dataset_env[candidate["dataset"]][candidate["env_name"]].append(candidate)
        available_counts[candidate["dataset"]] += 1

    quotas = allocate_balanced_counts(total_groups, available_counts)

    selected = []
    leftovers = []
    for dataset in DATASET_ORDER:
        env_groups = by_dataset_env.get(dataset, {})
        env_names = sorted(
            env_groups.keys(),
            key=lambda env_name: (
                THEME_ENV_PRIORITY.index(env_name) if env_name in THEME_ENV_PRIORITY else 999,
                env_name,
            ),
        )
        for env_name in env_names:
            env_groups[env_name].sort(key=lambda item: (item["object_id"], item["target_file"]))
        quota = quotas.get(dataset, 0)
        while quota > 0 and any(env_groups.get(env_name) for env_name in env_names):
            for env_name in env_names:
                bucket = env_groups.get(env_name, [])
                if not bucket:
                    continue
                selected.append(bucket.pop(0))
                quota -= 1
                if quota == 0:
                    break
        for env_name in env_names:
            leftovers.extend(env_groups.get(env_name, []))

    if len(selected) < total_groups:
        leftovers.sort(
            key=lambda item: (
                item["selection_score"],
                item["object_id"],
                item["target_file"],
            ),
            reverse=True,
        )
        needed = total_groups - len(selected)
        selected.extend(leftovers[:needed])

    selected.sort(
        key=lambda item: (
            DATASET_ORDER.index(item["dataset"]) if item["dataset"] in DATASET_ORDER else 99,
            -item["selection_score"],
            item["object_id"],
            item["target_file"],
        )
    )
    return selected[:total_groups], quotas


def summarize_selected_envs(selected: list[dict]):
    counts = Counter()
    score_sums = Counter()
    for item in selected:
        counts[item["env_name"]] += 1
        score_sums[item["env_name"]] += item["selection_score"]
    rows = []
    for env_name, count in counts.most_common():
        rows.append(
            {
                "env_name": env_name,
                "count": count,
                "mean_selection_score": float(score_sums[env_name] / count),
            }
        )
    return rows


def localize_selected_manifest(selected_manifest: dict, args, contrast_root: Path):
    localize_root = resolve_repo_path(args.localize_root)
    images_root = localize_root / "images" / args.split
    lighting_root = localize_root / "lighting" / args.split
    images_root.mkdir(parents=True, exist_ok=True)
    lighting_root.mkdir(parents=True, exist_ok=True)

    localized_samples = []
    for index, sample in enumerate(selected_manifest.get("samples", []), start=1):
        object_id = sample["object_id"]
        image_object_dir = images_root / object_id
        lighting_ldr_dir = lighting_root / "LDR" / object_id
        hdr_bucket = Path(sample["target_lighting_hdr_path"]).parent.parent.name
        lighting_hdr_dir = lighting_root / hdr_bucket / object_id

        image_object_dir.mkdir(parents=True, exist_ok=True)
        lighting_ldr_dir.mkdir(parents=True, exist_ok=True)
        lighting_hdr_dir.mkdir(parents=True, exist_ok=True)

        src_input = Path(sample["input_path"])
        src_gt = Path(sample["gt_path"])
        src_rt = src_gt.parent / f"{int(sample['view_idx']):03d}_RT.npy"
        src_normals = src_gt.parent / f"{int(sample['view_idx']):03d}_normals.png"
        src_ldr = Path(sample["target_lighting_ldr_path"])
        src_hdr = Path(sample["target_lighting_hdr_path"])

        dst_input = image_object_dir / src_input.name
        dst_gt = image_object_dir / src_gt.name
        dst_rt = image_object_dir / src_rt.name
        dst_normals = image_object_dir / src_normals.name
        dst_ldr = lighting_ldr_dir / src_ldr.name
        dst_hdr = lighting_hdr_dir / src_hdr.name

        shutil.copy2(src_input, dst_input)
        shutil.copy2(src_gt, dst_gt)
        if src_rt.exists():
            shutil.copy2(src_rt, dst_rt)
        if src_normals.exists():
            shutil.copy2(src_normals, dst_normals)
        shutil.copy2(src_ldr, dst_ldr)
        shutil.copy2(src_hdr, dst_hdr)

        localized_sample = dict(sample)
        localized_sample["input_path"] = str(dst_input)
        localized_sample["gt_path"] = str(dst_gt)
        localized_sample["target_lighting_ldr_path"] = str(dst_ldr)
        localized_sample["target_lighting_hdr_path"] = str(dst_hdr)
        localized_samples.append(localized_sample)

        if index % 64 == 0:
            print(f"[localize] copied {index}/{len(selected_manifest['samples'])} samples", flush=True)

    localized_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(contrast_root / "manifests" / f"highlight_top{args.total_groups}_manifest.json"),
        "localize_root": str(localize_root),
        "samples": localized_samples,
    }
    localized_manifest_path = contrast_root / "manifests" / f"highlight_top{args.total_groups}_manifest_local.json"
    dump_json(localized_manifest_path, localized_manifest)
    return localized_manifest_path


def build_metrics_tables(metrics_payload: dict, assets_manifest: dict):
    sample_to_dataset = {sample["sample_key"]: sample.get("dataset", "unknown") for sample in assets_manifest.get("samples", [])}
    global_metric_names = [
        "full_psnr",
        "foreground_psnr",
        "highlight_psnr",
        "highlight_mask_iou",
        "highlight_rmse",
    ]
    highlight_metric_names = [
        "highlight_psnr",
        "highlight_rmse",
        "highlight_mask_iou",
        "highlight_area_abs_error",
        "highlight_saturated_ratio_abs_error",
    ]

    def mean_of(records, metric_name):
        values = []
        for record in records:
            value = record.get("metrics", {}).get(metric_name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            values.append(value)
        if not values:
            return None
        return float(sum(values) / len(values))

    global_rows = []
    highlight_rows = []
    for method_name, payload in metrics_payload.get("methods", {}).items():
        records = payload.get("samples", [])
        by_dataset = defaultdict(list)
        for record in records:
            by_dataset[sample_to_dataset.get(record["sample_key"], "unknown")].append(record)

        split_order = DATASET_ORDER + ["overall"]
        for split_name in split_order:
            if split_name == "overall":
                split_records = records
                split_label = "overall"
            else:
                split_records = by_dataset.get(split_name, [])
                split_label = split_name
            if not split_records:
                continue

            base_row = {"split": split_label, "method": method_name, "sample_count": len(split_records)}
            global_row = dict(base_row)
            highlight_row = dict(base_row)
            for metric_name in global_metric_names:
                global_row[metric_name] = mean_of(split_records, metric_name)
            for metric_name in highlight_metric_names:
                highlight_row[metric_name] = mean_of(split_records, metric_name)
            global_rows.append(global_row)
            highlight_rows.append(highlight_row)

    return global_rows, global_metric_names, highlight_rows, highlight_metric_names


def render_markdown_table(title: str, rows: list[dict], metric_names: list[str]):
    headers = ["split", "method", "n", *metric_names]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---", "---", "---:"] + ["---:"] * len(metric_names)) + " |")
    for row in rows:
        values = [row["split"], row["method"], str(row["sample_count"])]
        for metric_name in metric_names:
            value = row.get(metric_name)
            values.append("-" if value is None else f"{value:.6f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def render_csv(rows: list[dict], metric_names: list[str]):
    headers = ["split", "method", "sample_count", *metric_names]
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    lines.append("")
    return "\n".join(lines)


def choose_best_ours_method(metrics_payload: dict):
    candidates = []
    for method_name in OURS_METHODS:
        overall = metrics_payload.get("methods", {}).get(method_name, {}).get("overall", {})
        candidates.append(
            (
                float(overall.get("highlight_psnr", {}).get("mean") or float("-inf")),
                float(overall.get("foreground_psnr", {}).get("mean") or float("-inf")),
                float(overall.get("highlight_mask_iou", {}).get("mean") or float("-inf")),
                method_name,
            )
        )
    candidates.sort(reverse=True)
    return candidates[0][-1]


def safe_float(value, default_low=False):
    if value is None:
        return float("-inf") if default_low else float("inf")
    value = float(value)
    if not math.isfinite(value):
        return float("-inf") if default_low else float("inf")
    return value


def remap_sample_paths(sample: dict, old_root: Path, new_root: Path):
    old_root_str = str(old_root)
    new_root_str = str(new_root)

    def remap_value(value):
        if isinstance(value, str) and value.startswith(old_root_str):
            return new_root_str + value[len(old_root_str):]
        if isinstance(value, dict):
            return {key: remap_value(sub_value) for key, sub_value in value.items()}
        if isinstance(value, list):
            return [remap_value(item) for item in value]
        return value

    return {key: remap_value(value) for key, value in sample.items()}


def copy_best_assets(best_root: Path, contrast_assets_manifest: dict, metrics_payload: dict, best_method: str, best_per_dataset: int):
    contrast_samples = contrast_assets_manifest.get("samples", [])
    sample_map = {sample["sample_key"]: sample for sample in contrast_samples}

    ours_records = metrics_payload.get("methods", {}).get(best_method, {}).get("samples", [])
    by_dataset = defaultdict(list)
    for record in ours_records:
        sample = sample_map.get(record["sample_key"])
        if sample is None:
            continue
        dataset = sample.get("dataset", "unknown")
        by_dataset[dataset].append((record, sample))

    selected_samples = []
    ranking_rows = []
    copied_assets_root = best_root / "assets" / "all_samples"
    copied_assets_root.mkdir(parents=True, exist_ok=True)

    for dataset in DATASET_ORDER:
        items = by_dataset.get(dataset, [])
        items.sort(
            key=lambda item: (
                safe_float(item[0]["metrics"].get("highlight_psnr"), default_low=True),
                safe_float(item[0]["metrics"].get("foreground_psnr"), default_low=True),
                safe_float(item[0]["metrics"].get("highlight_mask_iou"), default_low=True),
                -safe_float(item[0]["metrics"].get("highlight_rmse"), default_low=False),
                safe_float(item[0]["metrics"].get("full_psnr"), default_low=True),
                item[0]["sample_key"],
            ),
            reverse=True,
        )
        for rank, (record, sample) in enumerate(items[:best_per_dataset], start=1):
            src_dir = Path(sample.get("sample_dir") or Path(sample["input_export"]).parent)
            dst_dir = copied_assets_root / src_dir.name
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            copied_sample = remap_sample_paths(sample, src_dir, dst_dir)
            copied_sample["sample_dir"] = str(dst_dir)
            selected_samples.append(copied_sample)
            ranking_rows.append(
                {
                    "dataset": dataset,
                    "rank": rank,
                    "sample_key": record["sample_key"],
                    "highlight_psnr": record["metrics"].get("highlight_psnr"),
                    "foreground_psnr": record["metrics"].get("foreground_psnr"),
                    "highlight_mask_iou": record["metrics"].get("highlight_mask_iou"),
                    "highlight_rmse": record["metrics"].get("highlight_rmse"),
                }
            )

    selected_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_assets_manifest": contrast_assets_manifest.get("source_manifest"),
        "selection_method": best_method,
        "samples": selected_samples,
    }
    dump_json(best_root / "selected_assets_manifest.json", selected_manifest)
    dump_json(best_root / "best_ours_rankings.json", {"method": best_method, "rows": ranking_rows})
    return selected_manifest


def main():
    args = parse_args()
    contrast_root = resolve_repo_path(args.contrast_root)
    best_root = resolve_repo_path(args.best_root)
    contrast_root.mkdir(parents=True, exist_ok=True)
    best_root.mkdir(parents=True, exist_ok=True)

    selection_manifest_path = contrast_root / "manifests" / f"highlight_top{args.total_groups}_manifest.json"
    selection_summary_path = contrast_root / "manifests" / "selection_summary.json"

    if args.skip_existing and selection_manifest_path.exists() and selection_summary_path.exists():
        selected_manifest = load_json(selection_manifest_path)
    else:
        candidates, env_counter, per_dataset = scan_candidates(args)
        selected_samples, quotas = select_top_candidates(candidates, args.total_groups)
        selected_manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(resolve_repo_path(args.source_root)),
            "selection_strategy": "balanced_round_robin_filename_theme_hdri",
            "split": args.split,
            "samples": selected_samples,
        }
        dump_json(selection_manifest_path, selected_manifest)
        dump_json(
            selection_summary_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(resolve_repo_path(args.source_root)),
                "total_candidate_groups": len(candidates),
                "selected_group_count": len(selected_samples),
                "dataset_candidate_counts": dict(per_dataset),
                "dataset_selection_quotas": quotas,
                "selected_counts_by_dataset": dict(Counter(sample["dataset"] for sample in selected_samples)),
                "top_selected_envs": summarize_selected_envs(selected_samples),
                "candidate_env_frequency": dict(env_counter.most_common()),
            },
        )

    local_manifest_path = contrast_root / "manifests" / f"highlight_top{args.total_groups}_manifest_local.json"
    if not (args.skip_existing and local_manifest_path.exists()):
        local_manifest_path = localize_selected_manifest(selected_manifest, args, contrast_root)

    proxy_output_root = contrast_root
    proxy_assets_manifest = proxy_output_root / "exported_assets_manifest.json"
    if not (args.skip_existing and proxy_assets_manifest.exists()):
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_proxy_crossdomain_comparison.py"),
                "--manifest",
                str(local_manifest_path),
                "--output-root",
                str(proxy_output_root),
            ]
        )

    detailed_json = contrast_root / "stats" / "detailed_highlight_metrics.json"
    detailed_md = contrast_root / "stats" / "detailed_highlight_metrics.md"
    if not (args.skip_existing and detailed_json.exists() and detailed_md.exists()):
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_highlight_metrics_on_assets_manifest.py"),
                "--assets-manifest",
                str(proxy_assets_manifest),
                "--methods",
                "baseline",
                "dilightnet",
                "rgbx",
                "ours",
                "ours_full",
                "--output-json",
                str(detailed_json),
                "--output-md",
                str(detailed_md),
                "--compute-lpips",
                "false",
                "--compute-ssim",
                "false",
                "--device",
                str(args.metrics_device),
            ]
        )

    metrics_payload = load_json(detailed_json)
    assets_manifest_payload = load_json(proxy_assets_manifest)

    tables_dir = contrast_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    global_rows, global_metric_names, highlight_rows, highlight_metric_names = build_metrics_tables(metrics_payload, assets_manifest_payload)
    (tables_dir / "global_quality_table.md").write_text(
        render_markdown_table("Global Quality Table", global_rows, global_metric_names),
        encoding="utf-8",
    )
    (tables_dir / "highlight_quality_table.md").write_text(
        render_markdown_table("Highlight Quality Table", highlight_rows, highlight_metric_names),
        encoding="utf-8",
    )
    (tables_dir / "global_quality_table.csv").write_text(
        render_csv(global_rows, global_metric_names),
        encoding="utf-8",
    )
    (tables_dir / "highlight_quality_table.csv").write_text(
        render_csv(highlight_rows, highlight_metric_names),
        encoding="utf-8",
    )

    best_method = choose_best_ours_method(metrics_payload)
    selected_best_manifest = copy_best_assets(best_root, assets_manifest_payload, metrics_payload, best_method, args.best_per_dataset)

    dump_json(
        best_root / "summary.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_contrast_root": str(contrast_root),
            "selected_assets_manifest": str(best_root / "selected_assets_manifest.json"),
            "best_method": best_method,
            "best_per_dataset": args.best_per_dataset,
            "selected_sample_count": len(selected_best_manifest["samples"]),
        },
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
            "--assets-manifest",
            str(best_root / "selected_assets_manifest.json"),
            "--output",
            str(best_root / "best_ours_panel_headers.png"),
            "--columns",
            "input_image",
            "method:baseline",
            f"method:{best_method}",
            "ground_truth",
            "target_lighting",
            "--tile-size",
            "180",
            "--padding",
            "14",
            "--header-height",
            "60",
            "--hide-row-labels",
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
            "--assets-manifest",
            str(best_root / "selected_assets_manifest.json"),
            "--output",
            str(best_root / "best_ours_vs_dilightnet_rgbx_headers.png"),
            "--columns",
            "input_image",
            "method:baseline",
            "method:dilightnet",
            "method:rgbx",
            f"method:{best_method}",
            "ground_truth",
            "target_lighting",
            "--tile-size",
            "180",
            "--padding",
            "14",
            "--header-height",
            "60",
            "--hide-row-labels",
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
            "--assets-manifest",
            str(best_root / "selected_assets_manifest.json"),
            "--output",
            str(best_root / "best_ours_panel_no_text.png"),
            "--columns",
            "input_image",
            "method:baseline",
            f"method:{best_method}",
            "ground_truth",
            "target_lighting",
            "--tile-size",
            "180",
            "--padding",
            "14",
            "--header-height",
            "60",
            "--no-text",
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
            "--assets-manifest",
            str(best_root / "selected_assets_manifest.json"),
            "--output",
            str(best_root / "best_ours_vs_dilightnet_rgbx_no_text.png"),
            "--columns",
            "input_image",
            "method:baseline",
            "method:dilightnet",
            "method:rgbx",
            f"method:{best_method}",
            "ground_truth",
            "target_lighting",
            "--tile-size",
            "180",
            "--padding",
            "14",
            "--header-height",
            "60",
            "--no-text",
        ]
    )

    readme_lines = [
        "# Highlight HDRI Contrast Benchmark",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- source_root: {resolve_repo_path(args.source_root)}",
        f"- split: {args.split}",
        f"- selected_group_count: {len(selected_manifest['samples'])}",
        f"- contrast_root: {contrast_root}",
        f"- best_root: {best_root}",
        f"- best_ours_method: {best_method}",
        f"- compared_proxy_methods: {', '.join(CONTRAST_METHODS + OURS_METHODS)}",
        "",
        "## Notes",
        "",
        "- This benchmark currently runs in `proxy` generation mode because direct large-scale model inference is still unstable on the `/4T` weight-loading path.",
        f"- Selection uses filename-themed highlight HDRIs: {', '.join(THEME_ENV_PRIORITY)}.",
        f"- Selected samples are localized under {resolve_repo_path(args.localize_root)} before proxy generation.",
        "- Raw comparison artifacts and metrics are stored under `effects/contrast`.",
        "- The best-performing OURS subset is copied under `effects/best`.",
        "",
        "## Key Files",
        "",
        f"- selection_manifest: {selection_manifest_path}",
        f"- localized_manifest: {local_manifest_path}",
        f"- contrast_assets_manifest: {proxy_assets_manifest}",
        f"- detailed_metrics_json: {detailed_json}",
        f"- global_quality_table_md: {tables_dir / 'global_quality_table.md'}",
        f"- highlight_quality_table_md: {tables_dir / 'highlight_quality_table.md'}",
        f"- best_selected_assets_manifest: {best_root / 'selected_assets_manifest.json'}",
        f"- best_ours_panel_headers: {best_root / 'best_ours_panel_headers.png'}",
        f"- best_ours_panel_no_text: {best_root / 'best_ours_panel_no_text.png'}",
        f"- best_ours_vs_dilightnet_rgbx_headers: {best_root / 'best_ours_vs_dilightnet_rgbx_headers.png'}",
        f"- best_ours_vs_dilightnet_rgbx_no_text: {best_root / 'best_ours_vs_dilightnet_rgbx_no_text.png'}",
        "",
    ]
    (contrast_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    (best_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"wrote {contrast_root / 'README.md'}", flush=True)
    print(f"wrote {best_root / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
