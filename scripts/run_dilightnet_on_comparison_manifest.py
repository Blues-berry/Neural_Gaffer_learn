import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.foreground_mask_utils import fallback_white_background_mask, load_image_array, resolve_foreground_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiLightNet on comparison-manifest samples.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True, help="Output root in <root>/<object_id>/pred_image/<target_file> layout.")
    parser.add_argument("--staging-dir", default="logs/relighting_comparison/dilightnet_staging")
    parser.add_argument("--dilightnet-repo", default="external/DiLightNet_full")
    parser.add_argument("--python-bin", default=None, help="Python executable for the DiLightNet environment.")
    parser.add_argument("--foreground-background-threshold", type=float, default=0.96)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--run-official", action="store_true", help="Actually invoke DiLightNet official infer_img.py")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def load_mask(object_dir: Path, view_idx: int, gt_path: Path, background_threshold: float):
    mask, _ = resolve_foreground_mask(str(object_dir), view_idx=view_idx, reference_image_path=str(gt_path))
    if mask is None:
        rgb = load_image_array(str(gt_path))
        mask = fallback_white_background_mask(rgb, background_threshold=background_threshold)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.clip(mask, 0.0, 1.0)


def write_masked_provisional(input_path: Path, mask: np.ndarray, out_image: Path, out_mask: Path):
    image = Image.open(input_path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((512, 512), Image.Resampling.BILINEAR)

    rgb = np.asarray(image, dtype=np.uint8)
    alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
    white = np.full_like(rgb, 255)
    masked = (rgb.astype(np.float32) * alpha[..., None] + white.astype(np.float32) * (1.0 - alpha[..., None])).clip(0, 255).astype(np.uint8)

    Image.fromarray(masked, mode="RGB").save(out_image)
    mask_img.save(out_mask)


def stage_sample(sample: dict, staging_root: Path, background_threshold: float):
    gt_path = Path(sample["gt_path"])
    object_dir = gt_path.parent
    mask = load_mask(object_dir, int(sample["view_idx"]), gt_path, background_threshold)

    sample_stage = staging_root / f"{sample['preset']}_{sample['object_id']}_v{sample['view_idx']:03d}_t{sample['target_lighting_index']:03d}"
    sample_stage.mkdir(parents=True, exist_ok=True)
    prov_img = sample_stage / "prov_img.png"
    mask_path = sample_stage / "mask.png"
    write_masked_provisional(Path(sample["input_path"]), mask, prov_img, mask_path)

    metadata = {
        "object_id": sample["object_id"],
        "target_file": sample["target_file"],
        "target_lighting_hdr_path": sample["target_lighting_hdr_path"],
        "target_lighting_ldr_path": sample["target_lighting_ldr_path"],
        "prov_img": str(prov_img),
        "mask_path": str(mask_path),
    }
    (sample_stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sample_stage, metadata


def expected_dilightnet_output(repo_dir: Path, prov_img: Path, env_map: Path):
    prov_img_id = prov_img.stem
    lighting_id = f"env_map-{env_map.stem}"
    frame_num_id = "frames-1"
    return repo_dir / "tmp" / prov_img_id / lighting_id / frame_num_id / "relighting00_0.png"


def run_official_inference(repo_dir: Path, python_bin: str, sample_stage: Path, metadata: dict, prompt: str, steps: int, cfg: float, seed: int):
    prov_img = Path(metadata["prov_img"]).resolve()
    mask_path = Path(metadata["mask_path"]).resolve()
    env_map = Path(metadata["target_lighting_hdr_path"]).resolve()

    command = [
        python_bin,
        "infer_img.py",
        "--prov_img",
        str(prov_img),
        "--mask_path",
        str(mask_path),
        "--env_map",
        str(env_map),
        "--frames",
        "1",
        "--num_imgs_per_prompt",
        "1",
        "--prompt",
        prompt,
        "--steps",
        str(steps),
        "--cfg",
        str(cfg),
        "--seed",
        str(seed),
        "--nouse_sam",
    ]

    subprocess.run(command, cwd=repo_dir, check=True)
    output_path = expected_dilightnet_output(repo_dir, prov_img, env_map)
    if not output_path.exists():
        raise FileNotFoundError(f"DiLightNet finished but expected output was not found: {output_path}")
    return output_path


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(args.staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.dilightnet_repo)

    results = []
    for sample in manifest["samples"]:
        sample_stage, metadata = stage_sample(sample, staging_root, args.foreground_background_threshold)
        record = {
            "object_id": sample["object_id"],
            "target_file": sample["target_file"],
            "staging_dir": str(sample_stage),
            "prov_img": metadata["prov_img"],
            "mask_path": metadata["mask_path"],
            "env_map": metadata["target_lighting_hdr_path"],
        }

        if args.run_official:
            if args.python_bin is None:
                raise ValueError("--python-bin is required when --run-official is enabled")
            pred_path = run_official_inference(
                repo_dir=repo_dir.resolve(),
                python_bin=args.python_bin,
                sample_stage=sample_stage,
                metadata=metadata,
                prompt=args.prompt,
                steps=args.steps,
                cfg=args.cfg,
                seed=args.seed,
            )
            out_dir = output_root / sample["object_id"] / "pred_image"
            out_dir.mkdir(parents=True, exist_ok=True)
            final_path = out_dir / sample["target_file"]
            shutil.copy2(pred_path, final_path)
            record["prediction"] = str(final_path)
            print(f"saved {final_path}")
        else:
            command_preview = {
                "cwd": str(repo_dir.resolve()),
                "python_bin": args.python_bin or "<dilightnet-python>",
                "prov_img": metadata["prov_img"],
                "mask_path": metadata["mask_path"],
                "env_map": metadata["target_lighting_hdr_path"],
                "target_output": str(output_root / sample["object_id"] / "pred_image" / sample["target_file"]),
            }
            record["command_preview"] = command_preview
            print(f"staged {sample_stage}")

        results.append(record)

        if not args.keep_staging and args.run_official:
            shutil.rmtree(sample_stage, ignore_errors=True)

    summary_path = output_root / "dilightnet_manifest_predictions.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
