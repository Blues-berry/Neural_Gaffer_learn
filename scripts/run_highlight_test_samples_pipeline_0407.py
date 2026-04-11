import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ASSETS_ROOT = Path(
    os.environ.get(
        "NEURAL_GAFFER_ORIGINAL_ASSETS_ROOT",
        REPO_ROOT / "external_data" / "neural_gaffer_original",
    )
)
ORIGINAL_RENDER_SCRIPTS = Path(
    os.environ.get(
        "NEURAL_GAFFER_ORIGINAL_RENDER_SCRIPTS",
        ORIGINAL_ASSETS_ROOT / "rendering_pipeline" / "Objavarse_rendering",
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone highlight_test_samples pipeline: render raw outputs, preprocess lighting, "
            "build a same-batch one-HDRI manifest, then launch comparison and ablation."
        )
    )
    parser.add_argument(
        "--effects-root",
        default=str(REPO_ROOT / "effects/0407/highlight_test_samples_samebatch_v1"),
    )
    parser.add_argument(
        "--object-manifest",
        default=str(
            ORIGINAL_ASSETS_ROOT
            / "subdataset/standalone/highlight_test_samples/manifests/highlight_test_model_manifest.json"
        ),
    )
    parser.add_argument(
        "--raw-render-root",
        default=str(ORIGINAL_ASSETS_ROOT / "external_sources/render_raw/highlight_test_samples"),
    )
    parser.add_argument(
        "--lighting-source-root",
        default=str(ORIGINAL_ASSETS_ROOT / "objaverse_lighting_domains/ecommerce_product"),
    )
    parser.add_argument(
        "--lighting-training-root",
        default=str(ORIGINAL_ASSETS_ROOT / "training_data/lighting/training_lighting_data_highlight_test_samples"),
    )
    parser.add_argument(
        "--comparison-suite",
        default=str(REPO_ROOT / "configs/comparison_suites/validation_samebatch_0407_baseline_vs_ours.json"),
    )
    parser.add_argument(
        "--ablation-suite-source",
        default=str(REPO_ROOT / "configs/comparison_suites/validation_samebatch_0407_ablations_source.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--view-count", type=int, default=12)
    parser.add_argument("--lighting-per-view", type=int, default=16)
    parser.add_argument("--view-idx", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--ours-resolution", type=int, default=256)
    parser.add_argument("--max-resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--round-to-multiple", type=int, default=8)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-preprocess-lighting", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--with-ablation", action="store_true", default=True)
    parser.add_argument("--without-ablation", dest="with_ablation", action="store_false")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path | None = None):
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=cwd or REPO_ROOT, env=env, check=True)


def load_object_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("samples", "objects", "models", "items", "entries"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported object manifest payload at {path}")


def count_rendered_objects(raw_render_root: Path):
    if not raw_render_root.exists():
        return 0
    return sum(1 for path in raw_render_root.iterdir() if path.is_dir())


def main():
    args = parse_args()
    effects_root = Path(args.effects_root)
    effects_root.mkdir(parents=True, exist_ok=True)

    object_manifest = Path(args.object_manifest)
    raw_render_root = Path(args.raw_render_root)
    lighting_source_root = Path(args.lighting_source_root)
    lighting_training_root = Path(args.lighting_training_root)
    comparison_suite = Path(args.comparison_suite)
    ablation_suite_source = Path(args.ablation_suite_source)

    manifests_root = effects_root / "manifests"
    comparison_output_root = effects_root / "comparison"
    ablation_output_root = effects_root / "ablation"
    runtime_root = effects_root / "runtime"
    manifests_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    object_entries = load_object_manifest(object_manifest)
    expected_object_count = len(object_entries)

    if not args.skip_render:
        rendered_count = count_rendered_objects(raw_render_root)
        if rendered_count < expected_object_count:
            run_cmd(
                [
                    sys.executable,
                    "scripts/distribute-general-rendering.py",
                    "--workers-per-gpu",
                    str(args.workers_per_gpu),
                    "--output-dir",
                    str(raw_render_root),
                    "--lighting-dir",
                    str(lighting_source_root),
                    "--input-models-path",
                    str(object_manifest),
                    "--no-download-missing",
                    "--num-gpus",
                    str(args.num_gpus),
                ],
                cwd=ORIGINAL_RENDER_SCRIPTS,
            )

    if not args.skip_preprocess_lighting:
        run_cmd(
            [
                sys.executable,
                str(ORIGINAL_RENDER_SCRIPTS / "scripts" / "preprocess_environment_map.py"),
                "--img_dir",
                str(raw_render_root),
                "--output_dir",
                str(lighting_training_root),
                "--lighting_dir",
                str(lighting_source_root),
                "--input_json",
                str(object_manifest),
                "--num_workers",
                str(args.num_workers),
                "--total_view",
                str(args.view_count),
                "--lighting_per_view",
                str(args.lighting_per_view),
            ],
            cwd=ORIGINAL_RENDER_SCRIPTS,
        )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "create_samebatch_standalone_one_hdri_manifest.py"),
            "--object-manifest",
            str(object_manifest),
            "--raw-root",
            str(raw_render_root),
            "--lighting-root",
            str(lighting_training_root),
            "--output-dir",
            str(manifests_root),
            "--dataset-name",
            "highlight_test_samples",
            "--preset",
            "highlight_test_samebatch_onehdri",
            "--source-bucket",
            "standalone",
            "--image-split",
            "highlight_test",
            "--lighting-split",
            "highlight_test",
            "--view-idx",
            str(args.view_idx),
            "--shard-size",
            str(args.page_size),
        ]
    )

    comparison_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_validation_samebatch_pipeline_0407.py"),
        "--manifest-dir",
        str(manifests_root / "shards"),
        "--comparison-output-root",
        str(comparison_output_root),
        "--ablation-output-root",
        str(ablation_output_root),
        "--comparison-suite",
        str(comparison_suite),
        "--ablation-suite-source",
        str(ablation_suite_source),
        "--device",
        args.device,
        "--ours-resolution",
        str(args.ours_resolution),
        "--max-resolution",
        str(args.max_resolution),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--round-to-multiple",
        str(args.round_to_multiple),
        "--metrics-device",
        args.metrics_device,
        "--start-shard",
        "1",
        "--page-size",
        str(args.page_size),
        "--tile-size",
        str(args.tile_size),
        "--input-image-key",
        "white",
        "--method-image-key",
        "composited",
        "--ground-truth-image-key",
        "composited",
        "--visual-tag",
        "input_white_methods_gt_hdrbg",
        "--no-keep-input-resolution",
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--continue-on-error"] if args.continue_on_error else []),
        *(["--without-ablation"] if not args.with_ablation else []),
    ]
    run_cmd(comparison_cmd)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "effects_root": str(effects_root),
        "object_manifest": str(object_manifest),
        "raw_render_root": str(raw_render_root),
        "lighting_source_root": str(lighting_source_root),
        "lighting_training_root": str(lighting_training_root),
        "comparison_output_root": str(comparison_output_root),
        "ablation_output_root": str(ablation_output_root),
        "expected_object_count": expected_object_count,
        "with_ablation": bool(args.with_ablation),
    }
    (runtime_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
