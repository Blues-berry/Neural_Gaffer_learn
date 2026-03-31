import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a checkpoint suite and build a paper-style comparison panel.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--suite", required=True, help="JSON suite file with a `methods` list.")
    parser.add_argument("--pred-root", required=True, help="Directory that will contain per-method predictions.")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--panel-output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--foreground-background-threshold", type=float, default=0.96)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--header-height", type=int, default=54)
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


def load_suite(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload["methods"] if isinstance(payload, dict) else payload
    if not methods:
        raise ValueError(f"No methods found in {path}")
    return methods


def run_method(method: dict, args, output_root: Path):
    method_name = method["name"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_ours_on_comparison_manifest.py"),
        "--manifest",
        str(resolve_repo_path(args.manifest)),
        "--model-dir",
        str(resolve_repo_path(method["model_dir"])),
        "--output-root",
        str(output_root),
        "--method-name",
        method_name,
        "--device",
        args.device,
        "--resolution",
        str(args.resolution),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-inference-steps",
        str(args.num_inference_steps),
    ]

    checkpoint_path = resolve_repo_path(method.get("checkpoint_path"))
    if checkpoint_path is not None:
        cmd.extend(["--checkpoint-path", str(checkpoint_path)])

    pretrained_path = method.get("pretrained_model_name_or_path")
    if pretrained_path:
        cmd.extend(["--pretrained-model-name-or-path", str(resolve_repo_path(pretrained_path))])

    if method.get("enable_xformers"):
        cmd.append("--enable-xformers")

    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    print(f"[suite] running {method_name}: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


def main():
    args = parse_args()
    suite_path = resolve_repo_path(args.suite)
    methods = load_suite(suite_path)

    pred_root = resolve_repo_path(args.pred_root)
    assets_dir = resolve_repo_path(args.assets_dir)
    panel_output = resolve_repo_path(args.panel_output)
    pred_root.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    panel_output.parent.mkdir(parents=True, exist_ok=True)

    succeeded = []
    failed = []

    for method in methods:
        method_name = method["name"]
        output_root = pred_root / method_name
        summary_path = output_root / f"{method_name}_manifest_predictions.json"
        if args.skip_existing and summary_path.exists():
            print(f"[suite] skipping existing predictions for {method_name}: {summary_path}", flush=True)
            succeeded.append(method_name)
            continue

        output_root.mkdir(parents=True, exist_ok=True)
        returncode = run_method(method, args, output_root)
        if returncode == 0:
            succeeded.append(method_name)
        else:
            failed.append({"name": method_name, "returncode": returncode})
            print(f"[suite] method failed: {method_name} (returncode={returncode})", flush=True)
            if not args.continue_on_error:
                raise SystemExit(returncode)

    if not succeeded:
        raise SystemExit("[suite] no methods succeeded; aborting panel export")

    export_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_relighting_comparison_assets.py"),
        "--manifest",
        str(resolve_repo_path(args.manifest)),
        "--output-dir",
        str(assets_dir),
        "--foreground-background-threshold",
        str(args.foreground_background_threshold),
    ]
    for method_name in succeeded:
        export_cmd.extend(["--method-root", f"{method_name}={pred_root / method_name}"])

    print(f"[suite] exporting assets: {' '.join(export_cmd)}", flush=True)
    export_env = os.environ.copy()
    export_env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    subprocess.run(export_cmd, cwd=REPO_ROOT, check=True, env=export_env)

    assets_manifest = assets_dir / "exported_assets_manifest.json"
    panel_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
        "--assets-manifest",
        str(assets_manifest),
        "--output",
        str(panel_output),
        "--tile-size",
        str(args.tile_size),
        "--padding",
        str(args.padding),
        "--header-height",
        str(args.header_height),
        "--methods",
        *succeeded,
    ]
    print(f"[suite] building panel: {' '.join(panel_cmd)}", flush=True)
    panel_env = os.environ.copy()
    panel_env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    subprocess.run(panel_cmd, cwd=REPO_ROOT, check=True, env=panel_env)

    summary = {
        "manifest": str(resolve_repo_path(args.manifest)),
        "suite": str(suite_path),
        "pred_root": str(pred_root),
        "assets_dir": str(assets_dir),
        "panel_output": str(panel_output),
        "succeeded": succeeded,
        "failed": failed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "device": args.device,
    }
    summary_path = panel_output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[suite] wrote summary {summary_path}", flush=True)


if __name__ == "__main__":
    main()
