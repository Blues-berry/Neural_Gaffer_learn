import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision
from torchvision import transforms

from dataset.dataset_relighting_training import NeuralGafferTrainingData


def ensure_rank_zero():
    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return
    if not getattr(dist, "is_available", lambda: False)():
        return
    if not getattr(dist, "is_initialized", lambda: False)():
        dist.get_rank = lambda: 0


def build_transforms(resolution=256):
    return torchvision.transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def summarize_batch(batch):
    summary = {"tensor_shapes": {}, "non_tensor_keys": []}
    for key, value in batch.items():
        if torch.is_tensor(value):
            summary["tensor_shapes"][key] = list(value.shape)
        else:
            summary["non_tensor_keys"].append(key)
    return summary


def run_split(name, dataset, batch_size, max_batches, num_workers=0):
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    ok = True
    error = None
    batch_summary = None
    seen_batches = 0
    try:
        for idx, batch in enumerate(loader):
            if batch_summary is None:
                batch_summary = summarize_batch(batch)
            seen_batches += 1
            if seen_batches >= max_batches:
                break
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "split": name,
        "ok": ok,
        "batches_tested": seen_batches,
        "batch_summary": batch_summary,
        "error": error,
    }


def build_dataset_configs(ready_root, dataset_names):
    ready_root = Path(ready_root)
    configs = {}
    for dataset_name in dataset_names:
        dataset_dir = ready_root / dataset_name
        img_dir = dataset_dir / "images"
        lighting_dir = dataset_dir / "lighting"
        val_img_dir = dataset_dir / "val" / "images"
        val_lighting_dir = dataset_dir / "val" / "lighting"
        if not img_dir.exists() or not lighting_dir.exists():
            raise FileNotFoundError(f"Missing images/lighting for {dataset_name} at {dataset_dir}")
        if not val_img_dir.exists() or not val_lighting_dir.exists():
            raise FileNotFoundError(f"Missing val images/lighting for {dataset_name} at {dataset_dir}")
        configs[dataset_name] = {
            "img_dir": str(img_dir),
            "lighting_dir": str(lighting_dir),
            "val_img_dir": str(val_img_dir),
            "val_lighting_dir": str(val_lighting_dir),
        }
    return configs


def build_datasets(cfg, args, image_transforms):
    train_dataset = NeuralGafferTrainingData(
        img_dir=cfg["img_dir"],
        lighting_dir=cfg["lighting_dir"],
        image_transforms=image_transforms,
        lighting_per_view=args.train_lighting_per_view,
        total_view=args.train_total_view,
        validation=False,
        relighting_only=True,
        image_preprocessed=True,
        dataset_type="training_object_with_seen_envir",
        random_lighting_condition_prob=args.train_random_lighting_prob,
        foreground_background_threshold=args.foreground_background_threshold,
    )

    val_unseen_lighting = NeuralGafferTrainingData(
        img_dir=str(Path(cfg["val_img_dir"]) / "unseen_lighting"),
        lighting_dir=str(Path(cfg["val_lighting_dir"]) / "unseen_lighting"),
        image_transforms=image_transforms,
        lighting_per_view=args.val_lighting_per_view,
        total_view=args.val_total_view,
        validation=True,
        relighting_only=True,
        image_preprocessed=True,
        dataset_type="training_object_with_unseen_envir",
        foreground_background_threshold=args.foreground_background_threshold,
    )

    val_random_area = NeuralGafferTrainingData(
        img_dir=str(Path(cfg["val_img_dir"]) / "unseen_lighting"),
        lighting_dir=str(Path(cfg["val_lighting_dir"]) / "unseen_lighting"),
        image_transforms=image_transforms,
        lighting_per_view=args.val_lighting_per_view,
        total_view=args.val_total_view,
        validation=True,
        relighting_only=True,
        image_preprocessed=True,
        dataset_type="unseen_object_with_random_area_light_condition",
        foreground_background_threshold=args.foreground_background_threshold,
    )

    val_seen = NeuralGafferTrainingData(
        img_dir=str(Path(cfg["val_img_dir"]) / "seen_lighting"),
        lighting_dir=str(Path(cfg["val_lighting_dir"]) / "seen_lighting"),
        image_transforms=image_transforms,
        lighting_per_view=args.val_lighting_per_view,
        total_view=args.val_total_view,
        validation=True,
        relighting_only=True,
        image_preprocessed=True,
        dataset_type="unseen_object_with_seen_envir",
        foreground_background_threshold=args.foreground_background_threshold,
    )

    val_unseen = NeuralGafferTrainingData(
        img_dir=str(Path(cfg["val_img_dir"]) / "unseen_lighting"),
        lighting_dir=str(Path(cfg["val_lighting_dir"]) / "unseen_lighting"),
        image_transforms=image_transforms,
        lighting_per_view=args.val_lighting_per_view,
        total_view=args.val_total_view,
        validation=True,
        relighting_only=True,
        image_preprocessed=True,
        dataset_type="unseen_object_with_unseen_envir",
        foreground_background_threshold=args.foreground_background_threshold,
    )

    return {
        "train": train_dataset,
        "training_object_with_unseen_envir": val_unseen_lighting,
        "unseen_object_with_random_area_light_condition": val_random_area,
        "unseen_object_with_seen_envir": val_seen,
        "unseen_object_with_unseen_envir": val_unseen,
    }


def render_markdown(results):
    lines = []
    lines.append("# Dataset Validation Report")
    lines.append("")
    lines.append(f"Generated at: {results['generated_at_utc']}")
    lines.append("")
    lines.append("## Config")
    for key, value in results["config"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for dataset in results["datasets"]:
        lines.append(f"## {dataset['dataset']}")
        for split in dataset["splits"]:
            status = "OK" if split["ok"] else "FAIL"
            lines.append(f"- {split['split']}: {status} (batches={split['batches_tested']})")
            if split["error"]:
                lines.append(f"  error: {split['error']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_argparser():
    parser = argparse.ArgumentParser(description="Validate ready datasets by running a few batches per split.")
    parser.add_argument("--ready_root", type=str, default=str(REPO_ROOT / "logs" / "ready_subdatasets_20260328"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_total_view", type=int, default=12)
    parser.add_argument("--train_lighting_per_view", type=int, default=16)
    parser.add_argument("--train_random_lighting_prob", type=float, default=1.0)

    parser.add_argument("--val_total_view", type=int, default=4)
    parser.add_argument("--val_lighting_per_view", type=int, default=8)

    parser.add_argument("--foreground_background_threshold", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_argparser().parse_args()
    ensure_rank_zero()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ready_root = Path(args.ready_root)
    if not ready_root.exists():
        raise FileNotFoundError(f"Ready root not found: {ready_root}")

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted([p.name for p in ready_root.iterdir() if p.is_dir() and not p.name.startswith(".")])

    configs = build_dataset_configs(ready_root, dataset_names)
    image_transforms = build_transforms(args.resolution)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "logs" / f"dataset_validation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ready_root": str(ready_root),
            "datasets": dataset_names,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "train_total_view": args.train_total_view,
            "train_lighting_per_view": args.train_lighting_per_view,
            "train_random_lighting_prob": args.train_random_lighting_prob,
            "val_total_view": args.val_total_view,
            "val_lighting_per_view": args.val_lighting_per_view,
            "foreground_background_threshold": args.foreground_background_threshold,
        },
        "datasets": [],
    }

    for dataset_name in dataset_names:
        dataset_result = {
            "dataset": dataset_name,
            "splits": [],
        }
        datasets = build_datasets(configs[dataset_name], args, image_transforms)
        for split_name, dataset in datasets.items():
            dataset_result["splits"].append(
                run_split(
                    split_name,
                    dataset,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    num_workers=args.num_workers,
                )
            )
        results["datasets"].append(dataset_result)

    json_path = output_dir / "dataset_validation.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path = output_dir / "dataset_validation.md"
    md_path.write_text(render_markdown(results))

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote report: {md_path}")


if __name__ == "__main__":
    main()
