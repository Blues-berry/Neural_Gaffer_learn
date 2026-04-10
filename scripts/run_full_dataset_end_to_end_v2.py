import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full-dataset comparison, aggregate completed shards, and build grouped comparison panels."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ours-suite", default="configs/comparison_suites/ours_single_local_v2.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ours-resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-resolution", type=int, default=256)
    parser.add_argument("--round-to-multiple", type=int, default=8)
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--preserve-native-size", action="store_true", default=False)
    parser.add_argument("--no-preserve-native-size", dest="preserve_native_size", action="store_false")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--visual-tag", default="input_white_scene_bg")
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


def main():
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    aggregate_root = output_root / "aggregate"

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_full_dataset_comparison_v2.py"),
            "--manifest-dir",
            str(resolve_repo_path(args.manifest_dir)),
            "--output-root",
            str(output_root),
            "--ours-suite",
            str(resolve_repo_path(args.ours_suite)),
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
            "--start-shard",
            str(args.start_shard),
            *(["--limit-shards", str(args.limit_shards)] if args.limit_shards is not None else []),
            *(["--skip-existing"] if args.skip_existing else []),
            *(["--continue-on-error"] if args.continue_on_error else []),
            "--aggregate-output-root",
            str(aggregate_root),
            "--grouped-page-size",
            str(args.page_size),
            *(["--grouped-tile-size", str(args.tile_size)] if args.tile_size is not None else []),
            "--grouped-method-image-key",
            args.method_image_key,
            "--grouped-input-image-key",
            args.input_image_key,
            "--grouped-ground-truth-image-key",
            args.ground_truth_image_key,
            "--grouped-visual-tag",
            args.visual_tag,
            *(["--preserve-native-size"] if args.preserve_native_size else []),
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_sharded_comparison_results.py"),
            "--run-root",
            str(output_root),
            "--output-root",
            str(aggregate_root),
            "--page-size",
            str(args.page_size),
            *(["--tile-size", str(args.tile_size)] if args.tile_size is not None else []),
            "--build-grouped-panels",
            "--method-image-key",
            args.method_image_key,
            "--input-image-key",
            args.input_image_key,
            "--ground-truth-image-key",
            args.ground_truth_image_key,
            "--visual-tag",
            args.visual_tag,
            *(["--preserve-native-size"] if args.preserve_native_size else []),
        ]
    )


if __name__ == "__main__":
    main()
