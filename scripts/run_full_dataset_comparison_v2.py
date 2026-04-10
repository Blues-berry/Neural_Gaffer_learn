import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPETITORS = ("baseline", "dilightnet", "rgbx")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sharded full-dataset relighting comparison with proxy competitors and local OURS checkpoints."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ours-suite", default="configs/comparison_suites/ours_two_local_v2.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ours-resolution", type=int, default=256)
    parser.add_argument("--keep-input-resolution", action="store_true", default=False)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-resolution", type=int, default=512)
    parser.add_argument("--round-to-multiple", type=int, default=8)
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--competitor-methods", nargs="*", default=list(DEFAULT_COMPETITORS))
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-predictions", action="store_true", default=True)
    parser.add_argument("--no-skip-existing-predictions", dest="skip_existing_predictions", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--aggregate-at-end", action="store_true")
    parser.add_argument("--aggregate-after-each-shard", action="store_true", default=True)
    parser.add_argument("--no-aggregate-after-each-shard", dest="aggregate_after_each_shard", action="store_false")
    parser.add_argument("--aggregate-output-root", default=None)
    parser.add_argument("--grouped-page-size", type=int, default=5)
    parser.add_argument("--preserve-native-size", action="store_true", default=False)
    parser.add_argument("--no-preserve-native-size", dest="preserve_native_size", action="store_false")
    parser.add_argument("--grouped-tile-size", type=int, default=None)
    parser.add_argument("--grouped-method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--grouped-input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--grouped-ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--grouped-visual-tag", default="input_white_scene_bg")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_cmd(cmd: list[str], env: dict | None = None):
    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, check=True, env=env)


def load_suite(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods", payload)
    if not methods:
        raise ValueError(f"No methods defined in suite {path}")
    return methods


def method_summary_path(output_root: Path, method_name: str):
    return output_root / f"{method_name}_manifest_predictions.json"


def main():
    args = parse_args()
    manifest_dir = resolve_repo_path(args.manifest_dir)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    suite = load_suite(resolve_repo_path(args.ours_suite))
    aggregate_output_root = (
        resolve_repo_path(args.aggregate_output_root)
        if args.aggregate_output_root
        else output_root / "aggregate"
    )

    manifest_paths = sorted((manifest_dir / "shards").glob("manifest_shard_*.json"))
    manifest_paths = [path for path in manifest_paths if int(path.stem.split("_")[-1]) >= int(args.start_shard)]
    if args.limit_shards is not None:
        manifest_paths = manifest_paths[: max(int(args.limit_shards), 0)]
    if not manifest_paths:
        raise FileNotFoundError(f"No shard manifests found in {manifest_dir / 'shards'}")

    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

    for manifest_path in manifest_paths:
        shard_name = manifest_path.stem
        shard_root = output_root / "shards" / shard_name
        shard_root.mkdir(parents=True, exist_ok=True)
        print(f"[shard] {shard_name}", flush=True)

        try:
            method_roots = []
            for method in suite:
                method_name = method["name"]
                preds_root = shard_root / "preds" / method_name
                summary_path = method_summary_path(preds_root, method_name)
                if not (args.skip_existing and summary_path.exists()):
                    cmd = [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "run_ours_on_comparison_manifest_v2.py"),
                        "--manifest",
                        str(manifest_path),
                        "--model-dir",
                        str(resolve_repo_path(method["model_dir"])),
                        "--output-root",
                        str(preds_root),
                        "--method-name",
                        method_name,
                        "--device",
                        args.device,
                        "--resolution",
                        str(args.ours_resolution),
                        "--guidance-scale",
                        str(args.guidance_scale),
                        "--num-inference-steps",
                        str(args.num_inference_steps),
                        "--max-resolution",
                        str(args.max_resolution),
                        "--round-to-multiple",
                        str(args.round_to_multiple),
                    ]
                    if args.keep_input_resolution:
                        cmd.append("--keep-input-resolution")
                    else:
                        cmd.append("--no-keep-input-resolution")
                    checkpoint_path = method.get("checkpoint_path")
                    if checkpoint_path:
                        cmd.extend(["--checkpoint-path", str(resolve_repo_path(checkpoint_path))])
                    pretrained = method.get("pretrained_model_name_or_path")
                    if pretrained:
                        cmd.extend(["--pretrained-model-name-or-path", str(resolve_repo_path(pretrained))])
                    if method.get("enable_xformers"):
                        cmd.append("--enable-xformers")
                    if args.skip_existing_predictions:
                        cmd.append("--skip-existing-predictions")
                    run_cmd(cmd, env=env)
                method_roots.append((method_name, preds_root))

            competitor_root = shard_root / "proxy_competitors"
            competitor_preds_root = competitor_root / "preds"
            competitor_assets_summary = competitor_root / "stats" / "proxy_metrics_summary.json"
            if not (args.skip_existing and competitor_assets_summary.exists()):
                run_cmd(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "run_proxy_crossdomain_comparison.py"),
                        "--manifest",
                        str(manifest_path),
                        "--output-root",
                        str(competitor_root),
                        "--methods",
                        *args.competitor_methods,
                    ],
                    env=env,
                )

            assets_dir = shard_root / "assets"
            assets_manifest_path = assets_dir / "exported_assets_manifest.json"
            if not (args.skip_existing and assets_manifest_path.exists()):
                export_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "export_relighting_comparison_assets.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(assets_dir),
                ]
                for competitor in args.competitor_methods:
                    export_cmd.extend(["--method-root", f"{competitor}={competitor_preds_root / competitor}"])
                for method_name, method_root in method_roots:
                    export_cmd.extend(["--method-root", f"{method_name}={method_root}"])
                run_cmd(export_cmd, env=env)

            metrics_dir = shard_root / "metrics"
            metrics_json = metrics_dir / "highlight_metrics.json"
            metrics_md = metrics_dir / "highlight_metrics.md"
            metrics_csv = metrics_dir / "per_sample_metrics.csv"
            if not (args.skip_existing and metrics_json.exists() and metrics_csv.exists()):
                eval_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "evaluate_highlight_metrics_on_assets_manifest.py"),
                    "--assets-manifest",
                    str(assets_manifest_path),
                    "--methods",
                    *args.competitor_methods,
                    *[method["name"] for method in suite],
                    "--output-json",
                    str(metrics_json),
                    "--output-md",
                    str(metrics_md),
                    "--output-per-sample-csv",
                    str(metrics_csv),
                    "--compute-lpips",
                    "false",
                    "--compute-ssim",
                    "false",
                    "--device",
                    args.metrics_device,
                ]
                run_cmd(eval_cmd, env=env)

            if args.aggregate_after_each_shard:
                aggregate_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "aggregate_sharded_comparison_results.py"),
                    "--run-root",
                    str(output_root),
                    "--output-root",
                    str(aggregate_output_root),
                    "--page-size",
                    str(args.grouped_page_size),
                    *(["--tile-size", str(args.grouped_tile_size)] if args.grouped_tile_size is not None else []),
                    "--build-grouped-panels",
                    "--method-image-key",
                    args.grouped_method_image_key,
                    "--input-image-key",
                    args.grouped_input_image_key,
                    "--ground-truth-image-key",
                    args.grouped_ground_truth_image_key,
                    "--visual-tag",
                    args.grouped_visual_tag,
                ]
                if args.preserve_native_size:
                    aggregate_cmd.append("--preserve-native-size")
                run_cmd(aggregate_cmd, env=env)
        except Exception:
            if not args.continue_on_error:
                raise
            print(f"[warn] shard failed but continuing: {shard_name}", flush=True)

    if args.aggregate_at_end:
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "aggregate_sharded_comparison_results.py"),
                "--run-root",
                str(output_root),
                "--output-root",
                str(aggregate_output_root),
                "--page-size",
                str(args.grouped_page_size),
                *(["--tile-size", str(args.grouped_tile_size)] if args.grouped_tile_size is not None else []),
                "--build-grouped-panels",
                "--method-image-key",
                args.grouped_method_image_key,
                "--input-image-key",
                args.grouped_input_image_key,
                "--ground-truth-image-key",
                args.grouped_ground_truth_image_key,
                "--visual-tag",
                args.grouped_visual_tag,
                *(["--preserve-native-size"] if args.preserve_native_size else []),
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
