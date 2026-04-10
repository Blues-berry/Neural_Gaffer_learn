import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 0407 same-batch ablation end-to-end: inference, aggregate, sorted panels, and tables."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ours-resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-resolution", type=int, default=256)
    parser.add_argument("--round-to-multiple", type=int, default=8)
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--primary-method", default="officialval")
    parser.add_argument("--visual-tag", default="input_white_methods_gt_hdrbg")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_cmd(cmd: list[str]):
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, env=env, check=True)


def load_suite_methods(path: Path):
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods", payload)
    return [method["name"] for method in methods]


def main():
    args = parse_args()
    manifest_dir = resolve_repo_path(args.manifest_dir)
    output_root = resolve_repo_path(args.output_root)
    suite_path = resolve_repo_path(args.suite)
    aggregate_root = output_root / "aggregate"
    tables_root = aggregate_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_full_dataset_comparison_v2.py"),
            "--manifest-dir",
            str(manifest_dir),
            "--output-root",
            str(output_root),
            "--ours-suite",
            str(suite_path),
            "--device",
            args.device,
            "--ours-resolution",
            str(args.ours_resolution),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--max-resolution",
            str(args.max_resolution),
            "--round-to-multiple",
            str(args.round_to_multiple),
            "--metrics-device",
            args.metrics_device,
            "--competitor-methods",
            "--start-shard",
            str(args.start_shard),
            *(["--limit-shards", str(args.limit_shards)] if args.limit_shards is not None else []),
            *(["--skip-existing"] if args.skip_existing else []),
            "--skip-existing-predictions",
            *(["--continue-on-error"] if args.continue_on_error else []),
            "--no-aggregate-after-each-shard",
        ]
    )

    method_names = load_suite_methods(suite_path)

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_suite_sharded_results.py"),
            "--run-root",
            str(output_root),
            "--output-root",
            str(aggregate_root),
            "--methods",
            *method_names,
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_sorted_suite_panels.py"),
            "--assets-manifest",
            str(aggregate_root / "exported_assets_manifest.json"),
            "--per-sample-csv",
            str(aggregate_root / "per_sample_metrics.csv"),
            "--output-root",
            str(aggregate_root / "grouped_panels"),
            "--methods",
            *method_names,
            "--primary-method",
            args.primary_method,
            "--page-size",
            str(args.page_size),
            "--tile-size",
            str(args.tile_size),
            "--method-image-key",
            "composited",
            "--input-image-key",
            "white",
            "--ground-truth-image-key",
            "composited",
            "--visual-tag",
            args.visual_tag,
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_experiment_metric_tables.py"),
            "--metrics-summary",
            str(aggregate_root / "metrics_summary.json"),
            "--output-root",
            str(tables_root),
            "--methods",
            *method_names,
            "--title-prefix",
            "Ablation",
        ]
    )


if __name__ == "__main__":
    main()
