import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "NEURAL_GAFFER_MODEL_CACHE_ROOT",
        REPO_ROOT / "model_weights" / "neural_gaffer_model_cache",
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 0407 same-batch validation comparison first, then continue seamlessly with ablation experiments."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--comparison-output-root", required=True)
    parser.add_argument("--ablation-output-root", required=True)
    parser.add_argument(
        "--comparison-suite",
        default="configs/comparison_suites/validation_samebatch_0407_baseline_vs_ours.json",
    )
    parser.add_argument(
        "--ablation-suite-source",
        default="configs/comparison_suites/validation_samebatch_0407_ablations_source.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ours-resolution", type=int, default=256)
    parser.add_argument("--keep-input-resolution", action="store_true", default=False)
    parser.add_argument("--no-keep-input-resolution", dest="keep_input_resolution", action="store_false")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-resolution", type=int, default=512)
    parser.add_argument("--round-to-multiple", type=int, default=8)
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--visual-tag", default="input_white_methods_gt_hdrbg")
    parser.add_argument("--ablation-primary-method", default="officialval")
    parser.add_argument("--cache-root", default=str(DEFAULT_MODEL_CACHE_ROOT))
    parser.add_argument("--materialize-ablation-cache", action="store_true", default=True)
    parser.add_argument("--no-materialize-ablation-cache", dest="materialize_ablation_cache", action="store_false")
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods", payload)
    return [method["name"] for method in methods]


def main():
    args = parse_args()
    manifest_dir = resolve_repo_path(args.manifest_dir)
    comparison_output_root = resolve_repo_path(args.comparison_output_root)
    ablation_output_root = resolve_repo_path(args.ablation_output_root)
    comparison_output_root.mkdir(parents=True, exist_ok=True)
    ablation_output_root.mkdir(parents=True, exist_ok=True)

    comparison_suite = resolve_repo_path(args.comparison_suite)
    ablation_suite_source = resolve_repo_path(args.ablation_suite_source)
    ablation_suite_runtime = ablation_output_root / "configs" / "ablation_suite_runtime.json"

    if args.materialize_ablation_cache:
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "materialize_model_cache_from_suite.py"),
                "--suite",
                str(ablation_suite_source),
                "--output-suite",
                str(ablation_suite_runtime),
                "--cache-root",
                str(resolve_repo_path(args.cache_root)),
            ]
        )
    else:
        ablation_suite_runtime = ablation_suite_source

    common_args = [
        "--manifest-dir",
        str(manifest_dir),
        "--device",
        args.device,
        "--ours-resolution",
        str(args.ours_resolution),
        *(
            ["--keep-input-resolution"]
            if args.keep_input_resolution
            else ["--no-keep-input-resolution"]
        ),
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
        "--start-shard",
        str(args.start_shard),
        *(["--limit-shards", str(args.limit_shards)] if args.limit_shards is not None else []),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--continue-on-error"] if args.continue_on_error else []),
        "--grouped-page-size",
        str(args.page_size),
        "--grouped-tile-size",
        str(args.tile_size),
        "--grouped-method-image-key",
        args.method_image_key,
        "--grouped-input-image-key",
        args.input_image_key,
        "--grouped-ground-truth-image-key",
        args.ground_truth_image_key,
        "--grouped-visual-tag",
        args.visual_tag,
    ]

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_full_dataset_comparison_v2.py"),
            *common_args,
            "--output-root",
            str(comparison_output_root),
            "--ours-suite",
            str(comparison_suite),
            "--competitor-methods",
            "dilightnet",
            "rgbx",
            "--no-aggregate-after-each-shard",
            "--aggregate-at-end",
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_full_dataset_comparison_v2.py"),
            *common_args,
            "--output-root",
            str(ablation_output_root),
            "--ours-suite",
            str(ablation_suite_runtime),
            "--competitor-methods",
            "--no-aggregate-after-each-shard",
        ]
    )

    ablation_aggregate_root = ablation_output_root / "aggregate"
    ablation_methods = load_suite_methods(ablation_suite_runtime)

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_suite_sharded_results.py"),
            "--run-root",
            str(ablation_output_root),
            "--output-root",
            str(ablation_aggregate_root),
            "--methods",
            *ablation_methods,
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_sorted_suite_panels.py"),
            "--assets-manifest",
            str(ablation_aggregate_root / "exported_assets_manifest.json"),
            "--per-sample-csv",
            str(ablation_aggregate_root / "per_sample_metrics.csv"),
            "--output-root",
            str(ablation_aggregate_root / "grouped_panels"),
            "--methods",
            *ablation_methods,
            "--primary-method",
            args.ablation_primary_method,
            "--page-size",
            str(args.page_size),
            "--tile-size",
            str(args.tile_size),
            "--method-image-key",
            args.method_image_key,
            "--input-image-key",
            args.input_image_key,
            "--ground-truth-image-key",
            args.ground_truth_image_key,
            "--visual-tag",
            args.visual_tag,
        ]
    )


if __name__ == "__main__":
    main()
