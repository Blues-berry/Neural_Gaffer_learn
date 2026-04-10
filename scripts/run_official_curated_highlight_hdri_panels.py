import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BEST_OURS_CANDIDATES = ("ours", "ours_full")
COMPETITOR_METHODS = ("baseline", "dilightnet", "rgbx")
ALL_METHODS = COMPETITOR_METHODS + BEST_OURS_CANDIDATES
OBJECT_SEEDS = (
    "official_2000__0314e747909e4af387c8e7a4e239767c",
    "official_2000__00f192d718b64655928f38eeadb96478",
    "official_2000__014c656f5dd742c8acc126c232e11487",
    "official_2000__00619c9de6f14f03940b6cf72575d822",
)
OBJECT_POOL_PRIORITY = (
    *OBJECT_SEEDS,
    "official_2000__a07feda2faee4b74b249419dbd6c342c",
    "official_2000__03876ac53a75432f96b0ce3dc27a3559",
    "official_2000__0b77e51dea1447baac173a700a87022d",
    "official_2000__31a3e283d1dc43b4a1965315ca4288bd",
    "official_2000__232a95f5c4be4411abc1d2f7cc28965f",
    "official_2000__c1e72cde423245d8aafb4ccbb776592f",
    "official_2000__233587a0f4834a358fce682ecf1db68d",
    "official_2000__1c913c27a7a8429a881c1c4f65450315",
    "official_2000__c6902220316d4dd6ba9f0902debb3cd9",
    "official_2000__bcc1d38bdb2544de96d6521699e47db5",
    "official_2000__a6577ecc2b794666bf149a2f9e6ccf70",
    "official_2000__4a99954e04a94575834aa26babf02120",
    "official_2000__4ad04a925ccb4cfba58afbd89fbe4bb8",
    "official_2000__981f03e9b0294d779047c214fd4406d9",
    "official_2000__1aff6628ca4f4686ba5862686b51490e",
    "official_2000__2f8422301f55425eb47361ac6a382bc3",
    "official_2000__44d080c60eb54138af568d907c6da6a4",
    "official_2000__9d4cbdea31814a909acc34e0f2236b32",
    "official_2000__9db9f64214ac451b9b19b7be07d844d9",
    "official_2000__7febf28fb92a4189b71dd3796af78c51",
    "official_2000__e0d266b235214965b51002e299c41b03",
    "official_2000__d481a17270dc479184e71bf3e4ad8712",
    "official_2000__2b07a9cc3d2a46d0b18652f88b72ffb2",
    "official_2000__67692c284381485ba01c3cfaa30ca7d1",
)
ENV_PRIORITY = (
    "studio",
    "interior",
    "city",
    "courtyard",
    "HDR_040_Field",
    "117_hdrmaps_com_free_2K",
    "128_hdrmaps_com_free_2K",
    "012_hdrmaps_com_free_2K",
    "064_hdrmaps_com_free_2K",
    "125_hdrmaps_com_free_2K",
    "087_hdrmaps_com_free_2K",
)
ENV_PRIORITY_SCORE = {env_name: float(len(ENV_PRIORITY) - idx) for idx, env_name in enumerate(ENV_PRIORITY)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a curated official highlight-HDRI proxy experiment and build paginated white-bg/background panels."
    )
    parser.add_argument("--source-root", default="logs/dataset_validation_unions/all_ready_plus_official_20260403")
    parser.add_argument("--split", default="unseen_lighting")
    parser.add_argument("--output-root", default="effects/contrast/official_curated_highlight_hdri_v1")
    parser.add_argument("--localize-root", default="effects/tmp_local/official_curated_highlight_hdri_v1")
    parser.add_argument("--model-count", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--run-detailed-metrics", action="store_true")
    parser.add_argument("--full-official-pool", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_cmd(cmd: list[str], cwd: Path | None = None):
    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=cwd or REPO_ROOT, check=True)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_target_file(target_file: str):
    stem = Path(target_file).stem
    view_token, lighting_token, env_name = stem.split("_", 2)
    return int(view_token), int(lighting_token), env_name


def object_highlight_score(input_path: Path):
    rgba = np.asarray(Image.open(input_path).convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] > 0.05
    if not np.any(alpha):
        return float("-inf"), {}
    pixels = rgb[alpha]
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    sat = pixels.max(axis=1) - pixels.min(axis=1)
    bright_ratio = float(np.mean(pixels.max(axis=1) > 0.88))
    score = float(
        np.quantile(luma, 0.995)
        - np.quantile(luma, 0.65)
        + 0.55 * bright_ratio
        - 0.05 * float(np.mean(sat))
    )
    metrics = {
        "q995_luma": float(np.quantile(luma, 0.995)),
        "q65_luma": float(np.quantile(luma, 0.65)),
        "bright_ratio": bright_ratio,
        "sat_mean": float(np.mean(sat)),
    }
    return score, metrics


def lighting_highlight_score(env_name: str):
    return float(ENV_PRIORITY_SCORE.get(env_name, 0.0))


def scan_candidates(args):
    source_root = resolve_repo_path(args.source_root)
    images_root = source_root / "images" / args.split
    lighting_root = source_root / "lighting" / args.split

    object_candidates = {}
    if args.full_official_pool:
        candidate_names = sorted(p.name for p in images_root.iterdir() if p.is_dir() and p.name.startswith("official_2000__"))
    else:
        candidate_names = list(OBJECT_POOL_PRIORITY) if OBJECT_POOL_PRIORITY else sorted(p.name for p in images_root.iterdir() if p.is_dir())
    for object_name in candidate_names:
        object_dir = images_root / object_name
        if not object_dir.is_dir() or not object_dir.name.startswith("official_2000__"):
            continue
        input_path = object_dir / "random_lighting_000.png"
        if not input_path.exists():
            continue

        if args.full_official_pool:
            object_score, object_metrics = 0.0, {}
        else:
            object_score, object_metrics = object_highlight_score(input_path)
        env_samples = defaultdict(list)
        lighting_ldr_dir = lighting_root / "LDR" / object_dir.name
        if not lighting_ldr_dir.exists():
            continue

        for gt_path in sorted(object_dir.glob("000_*.png")):
            if gt_path.name == "000_normals.png":
                continue
            view_idx, target_idx, env_name = parse_target_file(gt_path.name)
            if env_name not in ENV_PRIORITY:
                continue
            target_ldr_path = lighting_ldr_dir / gt_path.name
            target_hdr_path = lighting_root / "HDR_rescaled" / object_dir.name / gt_path.name
            if not target_hdr_path.exists():
                target_hdr_path = lighting_root / "HDR_normalized" / object_dir.name / gt_path.name
            if not target_ldr_path.exists() or not target_hdr_path.exists():
                continue
            env_samples[env_name].append(
                {
                    "dataset": "official_2000",
                    "dataset_label": "office",
                    "object_id": object_dir.name,
                    "view_idx": view_idx,
                    "target_lighting_index": target_idx,
                    "target_file": gt_path.name,
                    "env_name": env_name,
                    "preset": "official_curated_highlight",
                    "input_mode": "random_lighting",
                    "input_path": str(input_path),
                    "gt_path": str(gt_path),
                    "target_lighting_ldr_path": str(target_ldr_path),
                    "target_lighting_hdr_path": str(target_hdr_path),
                    "cond_lighting_index": 0,
                    "image_split": args.split,
                    "lighting_split": args.split,
                    "mask_source": "preserved_from_target",
                    "object_score": object_score,
                    "object_metrics": object_metrics,
                    "lighting_score": lighting_highlight_score(env_name),
                }
            )

        env_samples = {env_name: rows for env_name, rows in env_samples.items() if rows}
        if not env_samples:
            continue
        object_candidates[object_dir.name] = {
            "object_score": object_score,
            "object_metrics": object_metrics,
            "available_envs": [env_name for env_name in ENV_PRIORITY if env_name in env_samples],
            "env_samples": env_samples,
        }

    return object_candidates


def select_objects(object_candidates: dict[str, dict], model_count: int):
    selected = []
    selected_set = set()
    for object_id in OBJECT_SEEDS:
        if object_id in object_candidates and object_id not in selected_set:
            selected.append(object_id)
            selected_set.add(object_id)

    ranked = []
    for object_id, payload in object_candidates.items():
        if object_id in selected_set:
            continue
        available_env_count = len(payload["available_envs"])
        metrics = payload["object_metrics"]
        bright_ratio = metrics.get("bright_ratio", 0.0)
        adjusted_score = float(payload["object_score"]) + 0.02 * available_env_count
        if bright_ratio > 0.35:
            adjusted_score -= 0.18 * (bright_ratio - 0.35)
        ranked.append((adjusted_score, object_id))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if model_count <= 0:
        model_count = len(object_candidates)
    for _, object_id in ranked:
        if len(selected) >= model_count:
            break
        selected.append(object_id)
        selected_set.add(object_id)
    return selected[:model_count]


def choose_env_sample(object_payload: dict, env_usage: Counter):
    best_choice = None
    for env_name in object_payload["available_envs"]:
        samples = sorted(
            object_payload["env_samples"][env_name],
            key=lambda item: (-item["lighting_score"], item["target_file"]),
        )
        candidate = samples[0]
        choice_key = (
            env_usage[env_name],
            ENV_PRIORITY.index(env_name),
            -candidate["lighting_score"],
            candidate["target_file"],
        )
        if best_choice is None or choice_key < best_choice[0]:
            best_choice = (choice_key, candidate)
    return None if best_choice is None else deepcopy(best_choice[1])


def build_selected_manifest(object_candidates: dict[str, dict], args):
    chosen_object_ids = select_objects(object_candidates, args.model_count)
    env_usage = Counter()
    selected_samples = []
    selection_rows = []
    for object_id in chosen_object_ids:
        sample = choose_env_sample(object_candidates[object_id], env_usage)
        if sample is None:
            continue
        env_usage[sample["env_name"]] += 1
        sample["selection_rank"] = len(selected_samples) + 1
        selected_samples.append(sample)
        selection_rows.append(
            {
                "selection_rank": sample["selection_rank"],
                "object_id": object_id,
                "env_name": sample["env_name"],
                "target_file": sample["target_file"],
                "object_score": object_candidates[object_id]["object_score"],
                "lighting_score": sample["lighting_score"],
                "available_envs": object_candidates[object_id]["available_envs"],
                "object_metrics": object_candidates[object_id]["object_metrics"],
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(resolve_repo_path(args.source_root)),
        "split": args.split,
        "selection_strategy": "seeded_official2000_highlight_score_balanced_env",
        "selected_model_count": len(selected_samples),
        "samples": selected_samples,
        "selection_rows": selection_rows,
        "selected_env_usage": dict(env_usage),
    }


def localize_manifest(selected_manifest: dict, args, output_root: Path):
    localize_root = resolve_repo_path(args.localize_root)
    images_root = localize_root / "images" / args.split
    lighting_root = localize_root / "lighting" / args.split
    images_root.mkdir(parents=True, exist_ok=True)
    lighting_root.mkdir(parents=True, exist_ok=True)

    localized_samples = []
    for sample in selected_manifest.get("samples", []):
        object_id = sample["object_id"]
        image_object_dir = images_root / object_id
        lighting_ldr_dir = lighting_root / "LDR" / object_id
        hdr_bucket = Path(sample["target_lighting_hdr_path"]).parent.parent.name
        lighting_hdr_dir = lighting_root / hdr_bucket / object_id

        image_object_dir.mkdir(parents=True, exist_ok=True)
        lighting_ldr_dir.mkdir(parents=True, exist_ok=True)
        lighting_hdr_dir.mkdir(parents=True, exist_ok=True)

        src_input = Path(sample["input_path"])
        src_gt = Path(sample["gt_path"])
        src_rt = src_gt.parent / f"{int(sample['view_idx']):03d}_RT.npy"
        src_normals = src_gt.parent / f"{int(sample['view_idx']):03d}_normals.png"
        src_ldr = Path(sample["target_lighting_ldr_path"])
        src_hdr = Path(sample["target_lighting_hdr_path"])

        dst_input = image_object_dir / src_input.name
        dst_gt = image_object_dir / src_gt.name
        dst_rt = image_object_dir / src_rt.name
        dst_normals = image_object_dir / src_normals.name
        dst_ldr = lighting_ldr_dir / src_ldr.name
        dst_hdr = lighting_hdr_dir / src_hdr.name

        shutil.copy2(src_input, dst_input)
        shutil.copy2(src_gt, dst_gt)
        if src_rt.exists():
            shutil.copy2(src_rt, dst_rt)
        if src_normals.exists():
            shutil.copy2(src_normals, dst_normals)
        shutil.copy2(src_ldr, dst_ldr)
        shutil.copy2(src_hdr, dst_hdr)

        localized = deepcopy(sample)
        localized["input_path"] = str(dst_input)
        localized["gt_path"] = str(dst_gt)
        localized["target_lighting_ldr_path"] = str(dst_ldr)
        localized["target_lighting_hdr_path"] = str(dst_hdr)
        localized_samples.append(localized)

    localized_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(output_root / "manifests" / "selected_manifest.json"),
        "localize_root": str(localize_root),
        "samples": localized_samples,
    }
    return localized_manifest


def choose_best_ours_method(metrics_payload: dict):
    candidates = []
    for method_name in BEST_OURS_CANDIDATES:
        overall = metrics_payload.get("methods", {}).get(method_name, {}).get("overall", {})
        candidates.append(
            (
                float(overall.get("highlight_psnr", {}).get("mean") or float("-inf")),
                float(overall.get("foreground_psnr", {}).get("mean") or float("-inf")),
                float(overall.get("highlight_mask_iou", {}).get("mean") or float("-inf")),
                -float(overall.get("highlight_rmse", {}).get("mean") or float("inf")),
                method_name,
            )
        )
    candidates.sort(reverse=True)
    return candidates[0][-1]


def sample_key_from_sample(sample: dict):
    if sample.get("sample_key"):
        return sample["sample_key"]
    return (
        f"{sample.get('preset', 'na')}_{sample.get('object_id', 'unknown')}"
        f"_v{int(sample.get('view_idx', 0)):03d}"
        f"_t{int(sample.get('target_lighting_index', 0)):03d}"
    )


def load_proxy_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_proxy_sample_outcomes(assets_manifest: dict, proxy_rows: list[dict], best_ours_method: str):
    metric_index = defaultdict(dict)
    for row in proxy_rows:
        metric_index[row["sample_key"]][row["method"]] = {
            "fg_rmse": float(row["fg_rmse"]),
            "fg_psnr": float(row["fg_psnr"]),
            "highlight_rmse": float(row["highlight_rmse"]),
        }

    outcomes = []
    failures = {name: [] for name in COMPETITOR_METHODS}
    for sample in assets_manifest.get("samples", []):
        sample_key = sample_key_from_sample(sample)
        methods = {
            method_name: metric_index.get(sample_key, {}).get(method_name)
            for method_name in COMPETITOR_METHODS + (best_ours_method,)
        }
        methods = {name: metrics for name, metrics in methods.items() if metrics is not None}
        if not methods:
            continue
        winner = min(
            methods.items(),
            key=lambda item: (
                float(item[1]["fg_rmse"]),
                float(item[1]["highlight_rmse"]),
                -float(item[1]["fg_psnr"]),
                item[0],
            ),
        )[0]
        outcome = {
            "sample_key": sample_key,
            "object_id": sample.get("object_id"),
            "env_name": sample.get("env_name"),
            "winner": winner,
            "best_ours_method": best_ours_method,
            "method_metrics": methods,
        }
        outcomes.append(outcome)
        if winner in failures:
            failures[winner].append(sample)
    return outcomes, failures


def sample_rank_tuple(sample_metrics: dict):
    return (
        float(sample_metrics.get("highlight_psnr") or float("-inf")),
        float(sample_metrics.get("foreground_psnr") or float("-inf")),
        float(sample_metrics.get("highlight_mask_iou") or float("-inf")),
        -float(sample_metrics.get("highlight_rmse") or float("inf")),
        float(sample_metrics.get("full_psnr") or float("-inf")),
    )


def build_metric_index(metrics_payload: dict):
    metric_index = defaultdict(dict)
    for method_name, payload in metrics_payload.get("methods", {}).items():
        for record in payload.get("samples", []):
            metric_index[record["sample_key"]][method_name] = record["metrics"]
    return metric_index


def build_sample_outcome_summary(assets_manifest: dict, metrics_payload: dict, best_ours_method: str):
    metric_index = build_metric_index(metrics_payload)
    outcomes = []
    failures = {name: [] for name in COMPETITOR_METHODS}
    for sample in assets_manifest.get("samples", []):
        sample_key = sample_key_from_sample(sample)
        methods = {method_name: metric_index.get(sample_key, {}).get(method_name) for method_name in COMPETITOR_METHODS + (best_ours_method,)}
        methods = {name: metrics for name, metrics in methods.items() if metrics is not None}
        if not methods:
            continue
        winner = max(methods.items(), key=lambda item: sample_rank_tuple(item[1]))[0]
        outcome = {
            "sample_key": sample_key,
            "object_id": sample.get("object_id"),
            "env_name": sample.get("env_name"),
            "winner": winner,
            "best_ours_method": best_ours_method,
            "method_metrics": methods,
        }
        outcomes.append(outcome)
        if winner in failures:
            failures[winner].append(sample)
    return outcomes, failures


def remap_for_visual_mode(sample: dict, visual_mode: str):
    mapped = deepcopy(sample)
    sample_dir = Path(sample["sample_dir"])
    if visual_mode == "white":
        mapped["input_export"] = sample.get("input_white_export") or sample.get("input_export")
        mapped["ground_truth_export"] = sample.get("ground_truth_white_export") or str(sample_dir / "ground_truth_white_bg.png")
        for method_name, method_payload in mapped.get("methods", {}).items():
            white_bg_path = Path(method_payload.get("white_bg") or sample_dir / f"{method_name}_white_bg.png")
            if white_bg_path.exists():
                method_payload["composited"] = str(white_bg_path)
    elif visual_mode == "hdrbg":
        if sample.get("input_composited_export"):
            mapped["input_export"] = sample["input_composited_export"]
        if sample.get("ground_truth_composited_export"):
            mapped["ground_truth_export"] = sample["ground_truth_composited_export"]
    else:
        raise ValueError(f"Unsupported visual mode: {visual_mode}")
    return mapped


def write_subset_manifest(samples: list[dict], output_path: Path, visual_mode: str, selection_name: str):
    remapped_samples = [remap_for_visual_mode(sample, visual_mode) for sample in samples]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_name": selection_name,
        "visual_mode": visual_mode,
        "samples": remapped_samples,
    }
    dump_json(output_path, payload)
    return output_path


def chunk_samples(samples: list[dict], page_size: int):
    for index in range(0, len(samples), page_size):
        yield index // page_size + 1, samples[index:index + page_size]


def build_pair_panels(samples: list[dict], selection_name: str, competitor: str, best_ours_method: str, output_root: Path, page_size: int):
    if not samples:
        return []
    panel_paths = []
    for visual_mode in ("white", "hdrbg"):
        panel_dir = output_root / "panels" / selection_name / visual_mode
        manifest_dir = output_root / "panel_manifests" / selection_name / visual_mode
        for page_number, page_samples in chunk_samples(samples, page_size):
            manifest_path = manifest_dir / f"{competitor}_vs_{best_ours_method}_page_{page_number:02d}.json"
            write_subset_manifest(
                page_samples,
                manifest_path,
                visual_mode=visual_mode,
                selection_name=selection_name,
            )
            output_path = panel_dir / f"{competitor}_vs_{best_ours_method}_page_{page_number:02d}.png"
            run_cmd(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
                    "--assets-manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--columns",
                    "input_image",
                    f"method:{competitor}",
                    f"method:{best_ours_method}",
                    "ground_truth",
                    "target_lighting",
                    "--tile-size",
                    "180",
                    "--padding",
                    "14",
                    "--header-height",
                    "60",
                    "--hide-row-labels",
                ]
            )
            panel_paths.append(str(output_path))
    return panel_paths


def build_all_method_panels(samples: list[dict], selection_name: str, best_ours_method: str, output_root: Path, page_size: int):
    if not samples:
        return []
    panel_paths = []
    for visual_mode in ("white", "hdrbg"):
        panel_dir = output_root / "panels" / selection_name / visual_mode
        manifest_dir = output_root / "panel_manifests" / selection_name / visual_mode
        for page_number, page_samples in chunk_samples(samples, page_size):
            manifest_path = manifest_dir / f"all_methods_page_{page_number:02d}.json"
            write_subset_manifest(
                page_samples,
                manifest_path,
                visual_mode=visual_mode,
                selection_name=selection_name,
            )
            output_path = panel_dir / f"all_methods_page_{page_number:02d}.png"
            run_cmd(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"),
                    "--assets-manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--columns",
                    "input_image",
                    "method:baseline",
                    "method:dilightnet",
                    "method:rgbx",
                    f"method:{best_ours_method}",
                    "ground_truth",
                    "target_lighting",
                    "--tile-size",
                    "180",
                    "--padding",
                    "14",
                    "--header-height",
                    "60",
                    "--hide-row-labels",
                ]
            )
            panel_paths.append(str(output_path))
    return panel_paths


def main():
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    proxy_root = output_root / "proxy"
    assets_root = output_root / "assets"
    stats_root = output_root / "stats"
    manifests_root = output_root / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    selection_manifest_path = manifests_root / "selected_manifest.json"
    local_manifest_path = manifests_root / "selected_manifest_local.json"
    selection_summary_path = manifests_root / "selection_summary.json"

    if args.skip_existing and selection_manifest_path.exists() and local_manifest_path.exists():
        selected_manifest = load_json(selection_manifest_path)
        localized_manifest = load_json(local_manifest_path)
    else:
        object_candidates = scan_candidates(args)
        selected_manifest = build_selected_manifest(object_candidates, args)
        localized_manifest = localize_manifest(selected_manifest, args, output_root)
        dump_json(selection_manifest_path, selected_manifest)
        dump_json(local_manifest_path, localized_manifest)
        dump_json(selection_summary_path, selected_manifest)

    proxy_manifest_summary = proxy_root / "stats" / "proxy_metrics_summary.json"
    proxy_per_sample_csv = proxy_root / "stats" / "proxy_metrics_per_sample.csv"
    if not (args.skip_existing and proxy_manifest_summary.exists()):
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_proxy_crossdomain_comparison.py"),
                "--manifest",
                str(local_manifest_path),
                "--output-root",
                str(proxy_root),
            ]
        )

    exported_assets_manifest = assets_root / "exported_assets_manifest.json"
    if not (args.skip_existing and exported_assets_manifest.exists()):
        export_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_relighting_comparison_assets.py"),
            "--manifest",
            str(local_manifest_path),
            "--output-dir",
            str(assets_root),
        ]
        for method_name in ALL_METHODS:
            export_cmd.extend(["--method-root", f"{method_name}={proxy_root / 'preds' / method_name}"])
        run_cmd(export_cmd)

    assets_manifest = load_json(exported_assets_manifest)
    proxy_summary = load_json(proxy_manifest_summary)
    proxy_rows = load_proxy_rows(proxy_per_sample_csv)

    detailed_json = stats_root / "detailed_highlight_metrics.json"
    detailed_md = stats_root / "detailed_highlight_metrics.md"
    if args.run_detailed_metrics and not (args.skip_existing and detailed_json.exists() and detailed_md.exists()):
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_highlight_metrics_on_assets_manifest.py"),
                "--assets-manifest",
                str(exported_assets_manifest),
                "--methods",
                *ALL_METHODS,
                "--output-json",
                str(detailed_json),
                "--output-md",
                str(detailed_md),
                "--compute-lpips",
                "false",
                "--compute-ssim",
                "false",
                "--device",
                str(args.metrics_device),
            ]
        )

    if args.run_detailed_metrics and detailed_json.exists():
        metrics_payload = load_json(detailed_json)
        best_ours_method = choose_best_ours_method(metrics_payload)
        sample_outcomes, failure_samples = build_sample_outcome_summary(assets_manifest, metrics_payload, best_ours_method)
        metric_source = "detailed_highlight_metrics"
    else:
        best_ours_method = proxy_summary["best_ours_method"]
        sample_outcomes, failure_samples = build_proxy_sample_outcomes(assets_manifest, proxy_rows, best_ours_method)
        metric_source = "proxy_metrics_per_sample"

    dump_json(
        stats_root / "sample_outcomes.json",
        {
            "best_ours_method": best_ours_method,
            "metric_source": metric_source,
            "samples": sample_outcomes,
        },
    )

    all_samples = assets_manifest.get("samples", [])
    panels_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_ours_method": best_ours_method,
        "selected_model_count": len(all_samples),
        "page_size": args.page_size,
        "all_samples": {},
        "failures": {},
    }

    panels_summary["all_samples"]["all_methods"] = build_all_method_panels(
        all_samples,
        selection_name="all_samples",
        best_ours_method=best_ours_method,
        output_root=output_root,
        page_size=args.page_size,
    )
    for competitor in COMPETITOR_METHODS:
        panels_summary["all_samples"][competitor] = build_pair_panels(
            all_samples,
            selection_name="all_samples",
            competitor=competitor,
            best_ours_method=best_ours_method,
            output_root=output_root,
            page_size=args.page_size,
        )
        panels_summary["failures"][competitor] = build_pair_panels(
            failure_samples.get(competitor, []),
            selection_name=f"failures_{competitor}",
            competitor=competitor,
            best_ours_method=best_ours_method,
            output_root=output_root,
            page_size=args.page_size,
        )

    dump_json(output_root / "panels_summary.json", panels_summary)

    readme_lines = [
        "# Official Curated Highlight HDRI Panels",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- source_root: {resolve_repo_path(args.source_root)}",
        f"- split: {args.split}",
        f"- selected_model_count: {len(all_samples)}",
        f"- best_ours_method: {best_ours_method}",
        f"- methods: {', '.join(ALL_METHODS)}",
        f"- proxy_root: {proxy_root}",
        f"- assets_root: {assets_root}",
        "",
        "## Notes",
        "",
        "- Samples come from `official_2000` and are selected to favor highlight-prone objects plus highlight-friendly HDRIs.",
        f"- HDRI priority: {', '.join(ENV_PRIORITY)}.",
        "- `white` panels keep the object results on white background.",
        "- `hdrbg` panels composite the input/prediction/ground-truth onto the current-view HDRI background.",
        "- Failure subsets contain samples where the winner is not the selected best OURS method.",
        "- This run is generated with the low-cost proxy comparison pipeline.",
        "",
        "## Key Files",
        "",
        f"- selection_manifest: {selection_manifest_path}",
        f"- localized_manifest: {local_manifest_path}",
        f"- exported_assets_manifest: {exported_assets_manifest}",
        f"- proxy_metrics_summary_json: {proxy_manifest_summary}",
        f"- proxy_metrics_per_sample_csv: {proxy_per_sample_csv}",
        f"- detailed_metrics_json: {detailed_json if args.run_detailed_metrics else 'skipped by default'}",
        f"- sample_outcomes_json: {stats_root / 'sample_outcomes.json'}",
        f"- panels_summary_json: {output_root / 'panels_summary.json'}",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"wrote {output_root / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
