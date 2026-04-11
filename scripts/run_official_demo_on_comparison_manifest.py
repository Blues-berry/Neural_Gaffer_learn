import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_REPO = Path(
    os.environ.get(
        "NEURAL_GAFFER_OFFICIAL_BASELINE_REPO",
        REPO_ROOT / "external" / "official_neural_gaffer_baseline",
    )
)
DEFAULT_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "NEURAL_GAFFER_OFFICIAL_CHECKPOINT_ROOT",
        REPO_ROOT / "model_weights" / "neural_gaffer_model_cache" / "neural_gaffer_training0316",
    )
)
DEFAULT_ZERO123_PATH = Path(
    "/4T/huggingface_cache/models--kxic--zero123-xl/snapshots/7d8aec2223b93e84eb26893d1e732e013523474b"
)
RUNTIME_SHIM_DIR = REPO_ROOT / "scripts" / "runtime_shims"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the official Neural Gaffer demo on a comparison manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True, help="Root directory that will contain staging, logs, runs, and renamed predictions.")
    parser.add_argument("--method-name", default="official-demo")
    parser.add_argument("--official-repo", default=str(DEFAULT_OFFICIAL_REPO))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--checkpoint-name", default="checkpoint-80000")
    parser.add_argument("--pretrained-model-name-or-path", default=str(DEFAULT_ZERO123_PATH))
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-validation-images", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sample_key(sample: dict):
    return f"{sample['preset']}_{sample['object_id']}_v{int(sample['view_idx']):03d}_t{int(sample['target_lighting_index']):03d}"


def stage_sample(sample: dict, output_root: Path):
    key = sample_key(sample)
    stage_root = output_root / "staging" / key
    input_dir = stage_root / "input_img"
    lighting_root = stage_root / "lighting"
    input_dir.mkdir(parents=True, exist_ok=True)
    lighting_root.mkdir(parents=True, exist_ok=True)

    input_path = resolve_path(sample["input_path"])
    target_ldr_path = resolve_path(sample["target_lighting_ldr_path"])
    target_hdr_path = resolve_path(sample["target_lighting_hdr_path"])
    lighting_name = Path(sample["target_file"]).stem

    staged_input_path = input_dir / f"{sample['object_id']}.png"
    shutil.copy2(input_path, staged_input_path)

    staged_lighting_root = lighting_root / lighting_name
    (staged_lighting_root / "LDR").mkdir(parents=True, exist_ok=True)
    (staged_lighting_root / "HDR_normalized").mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_ldr_path, staged_lighting_root / "LDR" / "0.png")
    shutil.copy2(target_hdr_path, staged_lighting_root / "HDR_normalized" / "0.png")

    return {
        "key": key,
        "lighting_name": lighting_name,
        "input_dir": input_dir,
        "lighting_root": lighting_root,
        "run_dir": output_root / "runs" / key,
        "log_path": output_root / "logs" / f"{key}.log",
    }


def resolve_predicted_file(run_dir: Path, object_id: str, lighting_name: str):
    pred_dir = run_dir / object_id / "pred_image"
    candidates = [
        pred_dir / f"{lighting_name}_000.png",
        pred_dir / f"{lighting_name}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    globbed = sorted(pred_dir.glob(f"{lighting_name}_*.png"))
    if globbed:
        return globbed[0]
    raise FileNotFoundError(f"Official demo output not found under {pred_dir} for lighting {lighting_name}")


def run_sample(args, sample: dict, output_root: Path, official_repo: Path, checkpoint_root: Path, pretrained_path: Path):
    staged = stage_sample(sample, output_root)
    target_out = output_root / "preds" / sample["object_id"] / "pred_image" / sample["target_file"]
    sample_out = output_root / "preds" / "_by_sample" / staged["key"] / "pred_image" / sample["target_file"]
    target_out.parent.mkdir(parents=True, exist_ok=True)
    sample_out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and target_out.exists() and sample_out.exists():
        return {
            "sample_key": staged["key"],
            "predicted_path": str(target_out),
            "sample_specific_path": str(sample_out),
            "skipped": True,
            "log_path": str(staged["log_path"]),
        }

    staged["run_dir"].mkdir(parents=True, exist_ok=True)
    staged["log_path"].parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(official_repo / "neural_gaffer_inference_real_data.py"),
        "--pretrained_model_name_or_path",
        str(pretrained_path),
        "--output_dir",
        str(checkpoint_root),
        "--mixed_precision",
        args.mixed_precision,
        "--resume_from_checkpoint",
        args.checkpoint_name,
        "--total_view",
        "1",
        "--lighting_per_view",
        "1",
        "--val_img_dir",
        str(staged["input_dir"]),
        "--val_lighting_dir",
        str(staged["lighting_root"]),
        "--save_dir",
        str(staged["run_dir"]),
        "--seed",
        str(args.seed),
        "--num_validation_images",
        str(args.num_validation_images),
        "--enable_xformers_memory_efficient_attention",
        "false",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    env["PYTHONPATH"] = (
        f"{RUNTIME_SHIM_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(RUNTIME_SHIM_DIR)
    )

    with staged["log_path"].open("w", encoding="utf-8") as log_file:
        return_code = subprocess.run(
            cmd,
            cwd=official_repo,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode

    if return_code != 0:
        raise RuntimeError(f"Official demo failed for {staged['key']} with return code {return_code}. See {staged['log_path']}")

    predicted = resolve_predicted_file(staged["run_dir"], sample["object_id"], staged["lighting_name"])
    shutil.copy2(predicted, target_out)
    shutil.copy2(predicted, sample_out)
    return {
        "sample_key": staged["key"],
        "predicted_path": str(target_out),
        "sample_specific_path": str(sample_out),
        "source_predicted_path": str(predicted),
        "log_path": str(staged["log_path"]),
        "skipped": False,
    }


def main():
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    output_root = resolve_path(args.output_root)
    official_repo = resolve_path(args.official_repo)
    checkpoint_root = resolve_path(args.checkpoint_root)
    pretrained_path = resolve_path(args.pretrained_model_name_or_path)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preds").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for sample in manifest["samples"]:
        result = run_sample(args, sample, output_root, official_repo, checkpoint_root, pretrained_path)
        results.append(result)

    summary = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "official_repo": str(official_repo),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_name": args.checkpoint_name,
        "method_name": args.method_name,
        "gpu_index": args.gpu_index,
        "mixed_precision": args.mixed_precision,
        "num_validation_images": args.num_validation_images,
        "samples": results,
    }
    summary_path = output_root / f"{args.method_name}_manifest_predictions.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
