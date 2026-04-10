import argparse
import csv
import json
import math
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

RUNS = OrderedDict(
    [
        ("abl00_base", Path("/4T/CXY/Neural_Gaffer/wandb/run-20260327_112126-c66bmmrj")),
        ("abl01_imgspace_fixed", Path("/4T/CXY/Neural_Gaffer/wandb/run-20260327_235824-dfsyx2cg")),
        ("abl02_quantile", Path("/4T/CXY/Neural_Gaffer/wandb/run-20260328_121118-4a9yd94d")),
        ("abl03_blur", Path("/4T/CXY/Neural_Gaffer/wandb/run-20260328_192945-plyikmtp")),
        ("abl04_relative", Path("/4T/CXY/Neural_Gaffer/wandb/run-20260329_065642-seyftjac")),
        ("abl05_full_main", Path("/4T/CXY/Neural_Gaffer/wandb/run-20260330_014457-2t3z4gz4")),
    ]
)

METHOD_LABELS = {
    "abl00_base": "Abl. Base",
    "abl01_imgspace_fixed": "Abl. ImgSpace Fixed",
    "abl02_quantile": "Abl. Quantile",
    "abl03_blur": "Abl. Blur",
    "abl04_relative": "Abl. Relative",
    "abl05_full_main": "Abl. Full Main",
}

SPLITS = OrderedDict(
    [
        ("unseen_object_with_unseen_envir", "uu"),
        ("unseen_object_with_seen_envir", "us"),
        ("unseen_object_with_random_area_light_condition", "ra"),
    ]
)

EXTRA_SPLITS = OrderedDict(
    [
        ("training_object_with_unseen_envir", "tu"),
        ("train", "train"),
    ]
)

PANEL_COLUMN_INDEX = {
    "input": 0,
    "gt": 1,
    "pred_0": 2,
    "pred_1": 3,
    "highlight_mask": 4,
    "highlight_weight": 5,
    "env_ldr": 6,
    "env_hdr": 7,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clean internal ablation tables and comparison panels from W&B validation panels."
    )
    parser.add_argument(
        "--output-root",
        default="/4T/CXY/Neural_Gaffer/effects/0407/validation_samebatch_onehdri_v1/internal_clean_ablation_from_wandb_v1",
    )
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_value(cfg: dict, key: str):
    value = cfg.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def mean(values):
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def fmt(value, digits: int = 4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def method_label(method_name: str):
    return METHOD_LABELS.get(method_name, method_name)


def load_run_payloads():
    payloads = OrderedDict()
    for method_name, run_dir in RUNS.items():
        cfg_path = run_dir / "files" / "config.yaml"
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not cfg_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing W&B files for {method_name}: {run_dir}")
        payloads[method_name] = {
            "run_dir": str(run_dir),
            "config": yaml.safe_load(cfg_path.read_text(encoding="utf-8")),
            "summary": load_json(summary_path),
        }
    return payloads


def build_metric_rows(payloads: OrderedDict):
    rows = []
    summary_payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "methods": {}}
    for method_name, payload in payloads.items():
        summary = payload["summary"]
        row = {
            "method": method_name,
            "label": method_label(method_name),
        }
        psnr_values = []
        ssim_values = []
        lpips_values = []
        highlight_mse_values = []
        highlight_ratio_values = []
        highlight_mse_ratio_values = []

        for split_name, short_name in SPLITS.items():
            row[f"psnr_{short_name}"] = summary.get(f"PSNR/{split_name}")
            row[f"ssim_{short_name}"] = summary.get(f"ssim_loss/{split_name}")
            row[f"lpips_{short_name}"] = summary.get(f"lpips_loss/{split_name}")
            row[f"highlight_mse_{short_name}"] = summary.get(f"highlight_mse/{split_name}")
            row[f"highlight_region_ratio_{short_name}"] = summary.get(f"highlight_region_ratio/{split_name}")
            row[f"highlight_mse_ratio_{short_name}"] = summary.get(f"highlight_mse_ratio/{split_name}")
            psnr_values.append(row[f"psnr_{short_name}"])
            ssim_values.append(row[f"ssim_{short_name}"])
            lpips_values.append(row[f"lpips_{short_name}"])
            highlight_mse_values.append(row[f"highlight_mse_{short_name}"])
            highlight_ratio_values.append(row[f"highlight_region_ratio_{short_name}"])
            highlight_mse_ratio_values.append(row[f"highlight_mse_ratio_{short_name}"])

        for split_name, short_name in EXTRA_SPLITS.items():
            row[f"psnr_{short_name}"] = summary.get(f"PSNR/{split_name}")
            if split_name != "train":
                row[f"ssim_{short_name}"] = summary.get(f"ssim_loss/{split_name}")
                row[f"lpips_{short_name}"] = summary.get(f"lpips_loss/{split_name}")
            else:
                row[f"ssim_{short_name}"] = summary.get("ssim_loss/train")
                row[f"lpips_{short_name}"] = summary.get("lpips_loss/train")

        row["psnr_mean"] = mean(psnr_values)
        row["ssim_mean"] = mean(ssim_values)
        row["lpips_mean"] = mean(lpips_values)
        row["highlight_mse_mean"] = mean(highlight_mse_values)
        row["highlight_region_ratio_mean"] = mean(highlight_ratio_values)
        row["highlight_mse_ratio_mean"] = mean(highlight_mse_ratio_values)
        rows.append(row)

        summary_payload["methods"][method_name] = {
            "psnr_mean": row["psnr_mean"],
            "ssim_mean": row["ssim_mean"],
            "lpips_mean": row["lpips_mean"],
            "highlight_mse_mean": row["highlight_mse_mean"],
            "highlight_region_ratio_mean": row["highlight_region_ratio_mean"],
            "highlight_mse_ratio_mean": row["highlight_mse_ratio_mean"],
            **{key: value for key, value in row.items() if key.startswith(("psnr_", "ssim_", "lpips_", "highlight_"))},
        }

    rows.sort(key=lambda item: (-float(item["psnr_mean"]), item["method"]))
    return rows, summary_payload


def render_global_md(rows):
    lines = [
        "# Internal Clean Ablation Main Metrics Table",
        "",
        "| method | PSNR Mean | PSNR (uu / us / ra) | PSNR (tu / train) | SSIM Mean | SSIM (uu / us / ra) | LPIPS Mean | LPIPS (uu / us / ra) |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["label"]
            + " | "
            + fmt(row["psnr_mean"])
            + " | "
            + f"{fmt(row['psnr_uu'])} / {fmt(row['psnr_us'])} / {fmt(row['psnr_ra'])}"
            + " | "
            + f"{fmt(row['psnr_tu'])} / {fmt(row['psnr_train'])}"
            + " | "
            + fmt(row["ssim_mean"])
            + " | "
            + f"{fmt(row['ssim_uu'])} / {fmt(row['ssim_us'])} / {fmt(row['ssim_ra'])}"
            + " | "
            + fmt(row["lpips_mean"])
            + " | "
            + f"{fmt(row['lpips_uu'])} / {fmt(row['lpips_us'])} / {fmt(row['lpips_ra'])}"
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_global_csv(rows):
    fieldnames = [
        "method",
        "label",
        "psnr_mean",
        "psnr_uu",
        "psnr_us",
        "psnr_ra",
        "psnr_tu",
        "psnr_train",
        "ssim_mean",
        "ssim_uu",
        "ssim_us",
        "ssim_ra",
        "lpips_mean",
        "lpips_uu",
        "lpips_us",
        "lpips_ra",
    ]
    lines = [",".join(fieldnames)]
    for row in rows:
        values = [row["method"], row["label"]] + [fmt(row[name]) for name in fieldnames[2:]]
        escaped = []
        for value in values:
            text = str(value)
            if "," in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            escaped.append(text)
        lines.append(",".join(escaped))
    lines.append("")
    return "\n".join(lines)


def render_highlight_md(rows):
    lines = [
        "# Internal Clean Ablation Highlight Metrics Table",
        "",
        "| method | H-MSE Mean | H-MSE (uu / us / ra) | H-Region Mean | H-Region (uu / us / ra) | H-MSE-Ratio Mean | H-MSE-Ratio (uu / us / ra) |",
        "| --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["label"]
            + " | "
            + fmt(row["highlight_mse_mean"])
            + " | "
            + f"{fmt(row['highlight_mse_uu'])} / {fmt(row['highlight_mse_us'])} / {fmt(row['highlight_mse_ra'])}"
            + " | "
            + fmt(row["highlight_region_ratio_mean"])
            + " | "
            + f"{fmt(row['highlight_region_ratio_uu'])} / {fmt(row['highlight_region_ratio_us'])} / {fmt(row['highlight_region_ratio_ra'])}"
            + " | "
            + fmt(row["highlight_mse_ratio_mean"])
            + " | "
            + f"{fmt(row['highlight_mse_ratio_uu'])} / {fmt(row['highlight_mse_ratio_us'])} / {fmt(row['highlight_mse_ratio_ra'])}"
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_highlight_csv(rows):
    fieldnames = [
        "method",
        "label",
        "highlight_mse_mean",
        "highlight_mse_uu",
        "highlight_mse_us",
        "highlight_mse_ra",
        "highlight_region_ratio_mean",
        "highlight_region_ratio_uu",
        "highlight_region_ratio_us",
        "highlight_region_ratio_ra",
        "highlight_mse_ratio_mean",
        "highlight_mse_ratio_uu",
        "highlight_mse_ratio_us",
        "highlight_mse_ratio_ra",
    ]
    lines = [",".join(fieldnames)]
    for row in rows:
        values = [row["method"], row["label"]] + [fmt(row[name]) for name in fieldnames[2:]]
        escaped = []
        for value in values:
            text = str(value)
            if "," in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            escaped.append(text)
        lines.append(",".join(escaped))
    lines.append("")
    return "\n".join(lines)


def build_module_rows(payloads: OrderedDict):
    rows = []
    for method_name, payload in payloads.items():
        cfg = payload["config"]
        rows.append(
            {
                "method": method_name,
                "label": method_label(method_name),
                "imgspace": "on" if safe_value(cfg, "use_image_space_highlight_loss") else "off",
                "imgspace_weight": safe_value(cfg, "image_space_constraint_weight"),
                "hlw": safe_value(cfg, "highlight_loss_weight"),
                "quantile": "on" if safe_value(cfg, "highlight_use_quantile_threshold") else "off",
                "highlight_quantile": safe_value(cfg, "highlight_quantile"),
                "highlight_min_threshold": safe_value(cfg, "highlight_min_threshold"),
                "highlight_max_threshold": safe_value(cfg, "highlight_max_threshold"),
                "blur_sigma": safe_value(cfg, "highlight_quantile_blur_sigma"),
                "relative_mode": safe_value(cfg, "highlight_relative_mode"),
                "kernel": safe_value(cfg, "highlight_local_kernel_size"),
                "foreground_threshold": safe_value(cfg, "foreground_background_threshold"),
                "random_lighting_prob": safe_value(cfg, "random_lighting_condition_prob"),
                "output_dir": safe_value(cfg, "output_dir"),
            }
        )
    return rows


def render_module_md(rows):
    lines = [
        "# Internal Clean Ablation Module Table",
        "",
        "| method | ImgSpace | ImgSpace W | HLW | Quantile | Q | Min | Max | Blur | Relative | Kernel | FG | Rand |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["label"]
            + " | "
            + str(row["imgspace"])
            + " | "
            + fmt(row["imgspace_weight"])
            + " | "
            + fmt(row["hlw"])
            + " | "
            + str(row["quantile"])
            + " | "
            + fmt(row["highlight_quantile"])
            + " | "
            + fmt(row["highlight_min_threshold"])
            + " | "
            + fmt(row["highlight_max_threshold"])
            + " | "
            + fmt(row["blur_sigma"])
            + " | "
            + str(row["relative_mode"])
            + " | "
            + fmt(row["kernel"], 0)
            + " | "
            + fmt(row["foreground_threshold"])
            + " | "
            + fmt(row["random_lighting_prob"])
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_module_csv(rows):
    fieldnames = [
        "method",
        "label",
        "imgspace",
        "imgspace_weight",
        "hlw",
        "quantile",
        "highlight_quantile",
        "highlight_min_threshold",
        "highlight_max_threshold",
        "blur_sigma",
        "relative_mode",
        "kernel",
        "foreground_threshold",
        "random_lighting_prob",
        "output_dir",
    ]
    lines = [",".join(fieldnames)]
    for row in rows:
        values = [row[name] for name in fieldnames]
        escaped = []
        for value in values:
            text = str(value)
            if "," in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            escaped.append(text)
        lines.append(",".join(escaped))
    lines.append("")
    return "\n".join(lines)


def extract_tiles(result_image_path: Path, tile_size: int):
    with Image.open(result_image_path) as image:
        width, height = image.size
        columns = width // tile_size
        if columns != len(PANEL_COLUMN_INDEX):
            raise RuntimeError(f"Unexpected result panel width in {result_image_path}: {image.size}")
        row_count = max(1, (height // tile_size))
        header_height = height - row_count * tile_size
        if header_height < 0:
            raise RuntimeError(f"Negative header height in {result_image_path}: {image.size}")
        return image.copy(), int(header_height), int(row_count)


def crop_result_tile(image: Image.Image, tile_size: int, header_height: int, row_index: int, column_name: str):
    column_index = PANEL_COLUMN_INDEX[column_name]
    left = column_index * tile_size
    top = header_height + row_index * tile_size
    return image.crop((left, top, left + tile_size, top + tile_size))


def collect_run_media(payloads: OrderedDict):
    media = OrderedDict()
    for method_name, payload in payloads.items():
        summary = payload["summary"]
        media[method_name] = {}
        for split_name in SPLITS.keys():
            result_entry = summary.get(f"{split_name}/result")
            if not result_entry:
                raise RuntimeError(f"Missing result entry for {method_name} {split_name}")
            filenames = [payload["run_dir"] + "/files/" + filename for filename in result_entry["filenames"]]
            media[method_name][split_name] = [Path(filename) for filename in filenames]
    return media


def build_sample_assets(output_root: Path, payloads: OrderedDict, tile_size: int):
    media = collect_run_media(payloads)
    anchor_method = "abl05_full_main"
    samples_by_split = OrderedDict((split_name, []) for split_name in SPLITS.keys())
    all_samples = []

    assets_root = output_root / "assets"
    for split_name, split_short in SPLITS.items():
        anchor_panels = media[anchor_method][split_name]
        for batch_idx, anchor_panel_path in enumerate(anchor_panels, start=1):
            anchor_panel, header_height, row_count = extract_tiles(anchor_panel_path, tile_size)
            for row_idx in range(row_count):
                sample_key = f"{split_short}_b{batch_idx:02d}_r{row_idx + 1:02d}"
                sample_dir = assets_root / split_short / sample_key
                sample_dir.mkdir(parents=True, exist_ok=True)

                input_tile = crop_result_tile(anchor_panel, tile_size, header_height, row_idx, "input")
                gt_tile = crop_result_tile(anchor_panel, tile_size, header_height, row_idx, "gt")
                target_tile = crop_result_tile(anchor_panel, tile_size, header_height, row_idx, "env_ldr")

                input_path = sample_dir / "input.png"
                gt_path = sample_dir / "ground_truth.png"
                target_path = sample_dir / "target_lighting.png"
                input_tile.save(input_path)
                gt_tile.save(gt_path)
                target_tile.save(target_path)

                method_entries = {}
                for method_name, split_media in media.items():
                    panel_path = split_media[split_name][batch_idx - 1]
                    panel_image, method_header_height, method_row_count = extract_tiles(panel_path, tile_size)
                    if method_header_height != header_height or method_row_count != row_count:
                        raise RuntimeError(f"Panel layout mismatch for {method_name} {split_name}")
                    pred_tile = crop_result_tile(panel_image, tile_size, method_header_height, row_idx, "pred_0")
                    pred_path = sample_dir / f"{method_name}.png"
                    pred_tile.save(pred_path)
                    method_entries[method_name] = {
                        "composited": str(pred_path),
                        "white_bg": str(pred_path),
                    }

                sample = {
                    "sample_key": sample_key,
                    "split": split_name,
                    "split_short": split_short,
                    "batch_index": batch_idx,
                    "row_index": row_idx + 1,
                    "input_export": str(input_path),
                    "input_white_export": str(input_path),
                    "input_composited_export": str(input_path),
                    "ground_truth_export": str(gt_path),
                    "ground_truth_white_export": str(gt_path),
                    "ground_truth_composited_export": str(gt_path),
                    "target_lighting_export": str(target_path),
                    "methods": method_entries,
                }
                samples_by_split[split_name].append(sample)
                all_samples.append(sample)
    write_json(
        output_root / "assets_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "wandb_validation_panels",
            "samples": all_samples,
        },
    )
    return samples_by_split


def chunk(items, page_size: int):
    for idx in range(0, len(items), page_size):
        yield idx // page_size + 1, items[idx: idx + page_size]


def run_cmd(cmd: list[str]):
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, check=True)


def write_panel_manifest(path: Path, samples: list[dict], selection_name: str):
    write_json(
        path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_name": selection_name,
            "samples": samples,
        },
    )


def build_panels(output_root: Path, samples_by_split: OrderedDict, page_size: int, tile_size: int):
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_size": page_size,
        "tile_size": tile_size,
        "methods": list(RUNS.keys()),
        "splits": {},
    }
    for split_name, split_short in SPLITS.items():
        split_samples = samples_by_split[split_name]
        panel_root = output_root / "panels" / "by_split" / split_short / "input_white_methods_gt_hdrbg"
        manifest_root = output_root / "panel_manifests" / "by_split" / split_short / "input_white_methods_gt_hdrbg"
        split_summary = {"sample_count": len(split_samples), "all_methods": [], "pair_panels": {}}

        for page_idx, page_samples in chunk(split_samples, page_size):
            manifest_path = manifest_root / f"all_methods_page_{page_idx:02d}.json"
            write_panel_manifest(manifest_path, page_samples, f"{split_short}_all_methods")
            output_path = panel_root / f"all_methods_page_{page_idx:02d}.png"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
                "--assets-manifest",
                str(manifest_path),
                "--output",
                str(output_path),
                "--columns",
                "input_image",
                *[f"method:{method_name}" for method_name in RUNS.keys()],
                "ground_truth",
                "target_lighting",
                "--method-image-key",
                "composited",
                "--input-image-key",
                "white",
                "--ground-truth-image-key",
                "composited",
                "--tile-size",
                str(tile_size),
                "--padding",
                "14",
                "--header-height",
                "60",
                "--hide-row-labels",
            ]
            run_cmd(cmd)
            run_cmd([*cmd, "--output", str(output_path.with_name(f"{output_path.stem}_headers.png"))])
            run_cmd([*cmd, "--output", str(output_path.with_name(f"{output_path.stem}_no_text.png")), "--no-text"])
            split_summary["all_methods"].append(str(output_path.with_name(f"{output_path.stem}_headers.png")))

        primary_method = "abl05_full_main"
        for method_name in RUNS.keys():
            if method_name == primary_method:
                continue
            split_summary["pair_panels"][method_name] = []
            for page_idx, page_samples in chunk(split_samples, page_size):
                manifest_path = manifest_root / f"{method_name}_vs_{primary_method}_page_{page_idx:02d}.json"
                write_panel_manifest(manifest_path, page_samples, f"{split_short}_{method_name}_vs_{primary_method}")
                output_path = panel_root / f"{method_name}_vs_{primary_method}_page_{page_idx:02d}.png"
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
                    "--assets-manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--columns",
                    "input_image",
                    f"method:{method_name}",
                    f"method:{primary_method}",
                    "ground_truth",
                    "target_lighting",
                    "--method-image-key",
                    "composited",
                    "--input-image-key",
                    "white",
                    "--ground-truth-image-key",
                    "composited",
                    "--tile-size",
                    str(tile_size),
                    "--padding",
                    "14",
                    "--header-height",
                    "60",
                    "--hide-row-labels",
                ]
                run_cmd(cmd)
                run_cmd([*cmd, "--output", str(output_path.with_name(f"{output_path.stem}_headers.png"))])
                run_cmd([*cmd, "--output", str(output_path.with_name(f"{output_path.stem}_no_text.png")), "--no-text"])
                split_summary["pair_panels"][method_name].append(str(output_path.with_name(f"{output_path.stem}_headers.png")))
        summary["splits"][split_short] = split_summary

    write_json(output_root / "panels_summary.json", summary)


def copy_key_figures(output_root: Path):
    key_root = output_root / "key_figures"
    key_root.mkdir(parents=True, exist_ok=True)
    for split_short in SPLITS.values():
        src = (
            output_root
            / "panels"
            / "by_split"
            / split_short
            / "input_white_methods_gt_hdrbg"
            / "all_methods_page_01_headers.png"
        )
        if src.exists():
            dst = key_root / f"{split_short}_all_methods_page_01_headers.png"
            dst.write_bytes(src.read_bytes())


def build_readme(rows, output_root: Path):
    best_main = max(rows, key=lambda item: float(item["psnr_mean"]))
    best_highlight = min(rows, key=lambda item: float(item["highlight_mse_mean"]))
    lines = [
        "# Internal Clean Ablation From W&B",
        "",
        "这套结果来自 clean ablation 六条 W&B 本地 run 的最终验证快照，不依赖当前盘上是否仍保留各自 checkpoint 目录。",
        "",
        "## 当前结论",
        "",
        f"- 主三项均值最佳：`{best_main['method']}` (`{best_main['label']}`), `PSNR Mean = {fmt(best_main['psnr_mean'])}`。",
        f"- 高光误差最优：`{best_highlight['method']}` (`{best_highlight['label']}`), `H-MSE Mean = {fmt(best_highlight['highlight_mse_mean'])}`。",
        "- 定性对比图按 split 分开整理，包含 `input / 各 ablation 版本 / ground-truth / target lighting`。",
        "",
        "## 入口",
        "",
        "- `tables/global_quality_table.md`",
        "- `tables/highlight_quality_table.md`",
        "- `tables/ablation_module_table.md`",
        "- `panels/by_split/uu/input_white_methods_gt_hdrbg`",
        "- `panels/by_split/us/input_white_methods_gt_hdrbg`",
        "- `panels/by_split/ra/input_white_methods_gt_hdrbg`",
        "",
    ]
    write_text(output_root / "README.md", "\n".join(lines))


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    payloads = load_run_payloads()
    rows, metrics_summary = build_metric_rows(payloads)
    module_rows = build_module_rows(payloads)

    tables_root = output_root / "tables"
    write_text(tables_root / "global_quality_table.md", render_global_md(rows))
    write_text(tables_root / "global_quality_table.csv", render_global_csv(rows))
    write_text(tables_root / "highlight_quality_table.md", render_highlight_md(rows))
    write_text(tables_root / "highlight_quality_table.csv", render_highlight_csv(rows))
    write_text(tables_root / "ablation_module_table.md", render_module_md(module_rows))
    write_text(tables_root / "ablation_module_table.csv", render_module_csv(module_rows))
    write_json(output_root / "metrics_summary.json", metrics_summary)

    samples_by_split = build_sample_assets(output_root, payloads, tile_size=args.tile_size)
    build_panels(output_root, samples_by_split, page_size=args.page_size, tile_size=args.tile_size)
    copy_key_figures(output_root)
    build_readme(rows, output_root)
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
