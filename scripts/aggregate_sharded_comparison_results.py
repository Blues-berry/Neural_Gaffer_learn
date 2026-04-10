import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OURS_METHODS = ("ours", "ours_full")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate sharded comparison assets/metrics into one global manifest and optional grouped panels."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--preserve-native-size", action="store_true")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--build-grouped-panels", action="store_true")
    parser.add_argument("--method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--visual-tag", default=None)
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


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def summarize_method_rows(rows: list[dict]):
    numeric_keys = set()
    for row in rows:
        for key, value in row.items():
            if key in {"sample_key", "preset", "object_id", "view_idx", "target_file", "mask_source", "method"}:
                continue
            if safe_float(value) is not None:
                numeric_keys.add(key)
    summary = {}
    for key in sorted(numeric_keys):
        values = [safe_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            summary[key] = float(sum(values) / len(values))
    return summary


def resolve_fg_rmse(summary: dict):
    value = safe_float(summary.get("fg_rmse"))
    if value is not None:
        return value
    value = safe_float(summary.get("foreground_rmse"))
    if value is not None:
        return value
    value = safe_float(summary.get("foreground_mse"))
    if value is not None:
        return float(value ** 0.5)
    return float("inf")


def main():
    args = parse_args()
    run_root = resolve_repo_path(args.run_root)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(path for path in (run_root / "shards").iterdir() if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under {run_root / 'shards'}")

    combined_samples = []
    combined_rows = []
    processed_shards = []
    methods_seen = Counter()

    for shard_dir in shard_dirs:
        assets_manifest_path = shard_dir / "assets" / "exported_assets_manifest.json"
        metrics_csv_path = shard_dir / "metrics" / "per_sample_metrics.csv"
        if not assets_manifest_path.exists() or not metrics_csv_path.exists():
            continue

        assets_payload = load_json(assets_manifest_path)
        combined_samples.extend(assets_payload.get("samples", []))

        with metrics_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        combined_rows.extend(rows)
        for row in rows:
            methods_seen[row.get("method")] += 1

        processed_shards.append(str(shard_dir))

    if not combined_samples or not combined_rows:
        raise RuntimeError("No completed shard results were found to aggregate.")

    combined_assets_manifest_path = output_root / "exported_assets_manifest.json"
    dump_json(
        combined_assets_manifest_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_root": str(run_root),
            "processed_shards": processed_shards,
            "samples": combined_samples,
        },
    )

    combined_csv_path = output_root / "per_sample_metrics.csv"
    fieldnames = list(combined_rows[0].keys())
    with combined_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(row)

    rows_by_method = defaultdict(list)
    for row in combined_rows:
        rows_by_method[row["method"]].append(row)

    overall_metrics = {
        method_name: summarize_method_rows(rows)
        for method_name, rows in sorted(rows_by_method.items())
    }
    available_ours = [method for method in OURS_METHODS if method in overall_metrics]
    best_ours_method = None
    if available_ours:
        best_ours_method = min(
            available_ours,
            key=lambda name: (
                resolve_fg_rmse(overall_metrics[name]),
                safe_float(overall_metrics[name].get("highlight_rmse")) or float("inf"),
                -(safe_float(overall_metrics[name].get("foreground_psnr")) or safe_float(overall_metrics[name].get("fg_psnr")) or float("-inf")),
            ),
        )

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_root": str(run_root),
        "processed_shard_count": len(processed_shards),
        "processed_shards": processed_shards,
        "sample_count": len(combined_samples),
        "metric_row_count": len(combined_rows),
        "methods": overall_metrics,
        "best_ours_method": best_ours_method,
        "combined_assets_manifest": str(combined_assets_manifest_path),
        "combined_per_sample_csv": str(combined_csv_path),
    }
    summary_path = output_root / "metrics_summary.json"
    dump_json(summary_path, summary_payload)

    if args.build_grouped_panels:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_grouped_highlight_panels.py"),
            "--assets-manifest",
            str(combined_assets_manifest_path),
            "--proxy-per-sample-csv",
            str(combined_csv_path),
            "--proxy-summary-json",
            str(summary_path),
            "--output-root",
            str(output_root / "grouped_panels"),
            "--page-size",
            str(args.page_size),
            *(["--tile-size", str(args.tile_size)] if args.tile_size is not None else []),
            "--method-image-key",
            args.method_image_key,
            "--input-image-key",
            args.input_image_key,
            "--ground-truth-image-key",
            args.ground_truth_image_key,
        ]
        if args.visual_tag:
            cmd.extend(["--visual-tag", args.visual_tag])
        if args.preserve_native_size:
            cmd.append("--preserve-native-size")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print(f"wrote {combined_assets_manifest_path}")
    print(f"wrote {combined_csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
