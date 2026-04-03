import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the minimal baseline-vs-ours highlight comparison experiment.")
    parser.add_argument("--suite", default="configs/comparison_suites/minimal_baseline_vs_ours_full_local.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--presets", nargs="*", default=["uu", "us", "ra"])
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--figure-samples-per-preset", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_str: str | None):
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_cmd(cmd: list[str], cwd: Path | None = None):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summarize_metric(records: list[dict], metric_name: str):
    values = []
    for record in records:
        value = record.get("metrics", {}).get(metric_name)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_method_records(metrics_payloads: dict[str, dict]):
    aggregated = defaultdict(list)
    for preset_name, payload in metrics_payloads.items():
        for method_name, method_payload in payload.get("methods", {}).items():
            for record in method_payload.get("samples", []):
                copied = dict(record)
                copied["preset"] = preset_name
                aggregated[method_name].append(copied)
    return aggregated


def aggregate_table_rows(aggregated_records: dict[str, list[dict]], metric_names: list[str]):
    rows = []
    for method_name, records in sorted(aggregated_records.items()):
        by_split = defaultdict(list)
        for record in records:
            by_split[record.get("preset", "unknown")].append(record)
        split_order = list(sorted(by_split.keys())) + ["overall"]
        for split_name in split_order:
            split_records = records if split_name == "overall" else by_split[split_name]
            row = {
                "split": split_name,
                "method": method_name,
                "sample_count": len(split_records),
            }
            for metric_name in metric_names:
                row[metric_name] = summarize_metric(split_records, metric_name)
            rows.append(row)
    return rows


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
        items = [str(row.get(header, "")) for header in headers]
        lines.append(",".join(items))
    return "\n".join(lines) + "\n"


def combine_assets_manifests(assets_paths: list[Path], output_path: Path):
    combined = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests": [str(path) for path in assets_paths],
        "samples": [],
    }
    for assets_path in assets_paths:
        payload = load_json(assets_path)
        combined["samples"].extend(payload.get("samples", []))
    dump_json(output_path, combined)
    return combined


def select_figure_samples(samples: list[dict], per_preset: int):
    selected = []
    by_preset = defaultdict(list)
    for sample in samples:
        by_preset[sample.get("preset", "unknown")].append(sample)
    for preset_name in ["uu", "us", "ra"]:
        selected.extend(by_preset.get(preset_name, [])[:per_preset])
    for preset_name in sorted(by_preset.keys()):
        if preset_name in {"uu", "us", "ra"}:
            continue
        selected.extend(by_preset[preset_name][:per_preset])
    return selected


def main():
    args = parse_args()
    suite_path = resolve_repo_path(args.suite)
    suite_payload = load_json(suite_path)
    method_names = [method["name"] for method in suite_payload.get("methods", [])]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = resolve_repo_path(args.output_root)
    if output_root is None:
        output_root = REPO_ROOT / "logs" / "relighting_comparison" / f"minimal_highlight_experiment_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifests_dir = output_root / "manifests"
    tables_dir = output_root / "tables"
    panels_dir = output_root / "panels"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    metrics_payloads = {}
    assets_paths = []

    for preset_name in args.presets:
        manifest_path = manifests_dir / f"{preset_name}_manifest.json"
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_relighting_comparison_manifest.py"),
                "--preset",
                preset_name,
                "--sample-count",
                str(args.sample_count),
                "--seed",
                str(args.seed),
                "--view-idx",
                str(args.view_idx),
                "--output",
                str(manifest_path),
            ]
        )

        pred_root = output_root / f"{preset_name}_preds"
        assets_dir = output_root / f"{preset_name}_assets"
        split_panel = output_root / f"{preset_name}_overview_panel.png"
        suite_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_checkpoint_panel_suite.py"),
            "--manifest",
            str(manifest_path),
            "--suite",
            str(suite_path),
            "--pred-root",
            str(pred_root),
            "--assets-dir",
            str(assets_dir),
            "--panel-output",
            str(split_panel),
            "--device",
            args.device,
            "--resolution",
            str(args.resolution),
            "--guidance-scale",
            str(args.guidance_scale),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--tile-size",
            "160",
            "--padding",
            "12",
            "--header-height",
            "54",
        ]
        if args.skip_existing:
            suite_cmd.append("--skip-existing")
        if args.continue_on_error:
            suite_cmd.append("--continue-on-error")
        run_cmd(suite_cmd)

        assets_manifest_path = assets_dir / "exported_assets_manifest.json"
        metrics_json = output_root / f"{preset_name}_highlight_metrics.json"
        metrics_md = output_root / f"{preset_name}_highlight_metrics.md"
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_highlight_metrics_on_assets_manifest.py"),
                "--assets-manifest",
                str(assets_manifest_path),
                "--methods",
                *method_names,
                "--output-json",
                str(metrics_json),
                "--output-md",
                str(metrics_md),
                "--device",
                args.device,
            ]
        )

        metrics_payloads[preset_name] = load_json(metrics_json)
        assets_paths.append(assets_manifest_path)

    combined_assets_path = output_root / "combined_assets_manifest.json"
    combined_assets = combine_assets_manifests(assets_paths, combined_assets_path)

    figure_samples = select_figure_samples(combined_assets.get("samples", []), per_preset=args.figure_samples_per_preset)
    figure_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_assets_manifest": str(combined_assets_path),
        "samples": figure_samples,
    }
    figure_manifest_path = output_root / "figure_assets_manifest.json"
    dump_json(figure_manifest_path, figure_manifest)

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
            "--assets-manifest",
            str(figure_manifest_path),
            "--output",
            str(panels_dir / "overview_panel.png"),
            "--columns",
            "input_image",
            "method:baseline",
            "method:ours_full",
            "ground_truth",
            "target_lighting",
            "--tile-size",
            "180",
            "--padding",
            "14",
            "--header-height",
            "60",
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
            "--assets-manifest",
            str(figure_manifest_path),
            "--output",
            str(panels_dir / "diagnostic_panel.png"),
            "--columns",
            "ground_truth",
            "gt_highlight_mask",
            "method:baseline",
            "method_mask:baseline",
            "method:ours_full",
            "method_mask:ours_full",
            "--tile-size",
            "180",
            "--padding",
            "14",
            "--header-height",
            "60",
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_highlight_zoom_panel.py"),
            "--assets-manifest",
            str(figure_manifest_path),
            "--output",
            str(panels_dir / "highlight_zoom_panel.png"),
            "--columns",
            "method:baseline",
            "method:ours_full",
            "ground_truth",
            "gt_highlight_mask",
            "method_mask:baseline",
            "method_mask:ours_full",
            "--methods",
            *method_names,
            "--focus-methods",
            *method_names,
            "--tile-size",
            "220",
            "--padding",
            "16",
            "--header-height",
            "64",
        ]
    )

    aggregated_records = aggregate_method_records(metrics_payloads)
    global_metric_names = [
        "full_psnr",
        "full_ssim",
        "lpips_full",
        "foreground_psnr",
        "foreground_ssim",
        "lpips_foreground",
    ]
    highlight_metric_names = [
        "highlight_psnr",
        "highlight_rmse",
        "highlight_mask_iou",
        "highlight_centroid_distance",
        "highlight_saturated_ratio_abs_error",
        "lpips_highlight_crop",
    ]
    global_rows = aggregate_table_rows(aggregated_records, global_metric_names)
    highlight_rows = aggregate_table_rows(aggregated_records, highlight_metric_names)

    global_md = render_markdown_table("Global Quality Table", global_rows, global_metric_names)
    highlight_md = render_markdown_table("Highlight Quality Table", highlight_rows, highlight_metric_names)
    (tables_dir / "global_quality_table.md").write_text(global_md, encoding="utf-8")
    (tables_dir / "highlight_quality_table.md").write_text(highlight_md, encoding="utf-8")
    (tables_dir / "global_quality_table.csv").write_text(render_csv(global_rows, global_metric_names), encoding="utf-8")
    (tables_dir / "highlight_quality_table.csv").write_text(render_csv(highlight_rows, highlight_metric_names), encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite_path),
        "method_names": method_names,
        "presets": args.presets,
        "sample_count_per_preset": args.sample_count,
        "total_samples": len(combined_assets.get("samples", [])),
        "device": args.device,
        "resolution": args.resolution,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "figure_samples_per_preset": args.figure_samples_per_preset,
        "artifacts": {
            "combined_assets_manifest": str(combined_assets_path),
            "figure_assets_manifest": str(figure_manifest_path),
            "overview_panel": str(panels_dir / "overview_panel.png"),
            "diagnostic_panel": str(panels_dir / "diagnostic_panel.png"),
            "highlight_zoom_panel": str(panels_dir / "highlight_zoom_panel.png"),
            "global_quality_table_md": str(tables_dir / "global_quality_table.md"),
            "highlight_quality_table_md": str(tables_dir / "highlight_quality_table.md"),
        },
        "notes": [
            "baseline uses the corrected historical 7cn19b1e checkpoint from /dev/shm.",
            "ours_full uses the strongest available local 80k checkpoint under logs/neural_gaffer_training_gpu1_highlight as a full-main-like local proxy.",
        ],
    }
    dump_json(output_root / "experiment_summary.json", summary)

    summary_md = [
        "# Minimal Highlight Experiment",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- suite: {summary['suite']}",
        f"- methods: {', '.join(method_names)}",
        f"- presets: {', '.join(args.presets)}",
        f"- sample_count_per_preset: {args.sample_count}",
        f"- total_samples: {summary['total_samples']}",
        f"- device: {args.device}",
        f"- overview_panel: {summary['artifacts']['overview_panel']}",
        f"- diagnostic_panel: {summary['artifacts']['diagnostic_panel']}",
        f"- highlight_zoom_panel: {summary['artifacts']['highlight_zoom_panel']}",
        f"- global_quality_table_md: {summary['artifacts']['global_quality_table_md']}",
        f"- highlight_quality_table_md: {summary['artifacts']['highlight_quality_table_md']}",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in summary["notes"]],
        "",
    ]
    (output_root / "README.md").write_text("\n".join(summary_md), encoding="utf-8")
    print(f"wrote {output_root / 'experiment_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
