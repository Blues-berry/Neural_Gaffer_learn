#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/4T/CXY/Neural_Gaffer_original")


@dataclass(frozen=True)
class DatasetSpec:
    status_json: Path
    raw_dir: Path
    active_patterns: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "official_2000": DatasetSpec(
        status_json=ROOT / "subdataset" / "official" / "subsets" / "official_2000" / "status.json",
        raw_dir=ROOT / "objaverse_jobs" / "official_2000" / "raw",
        active_patterns=(
            r"run_render_completion_daemon\.sh official_2000",
            r"run_official_preprocess_job\.sh",
            r"distribute-general-rendering\.py.*official_pending_1000\.json",
            r"preprocess_rendered_image\.py --img_dir /4T/CXY/Neural_Gaffer_original/objaverse_jobs/official_2000/raw",
            r"preprocess_environment_map\.py --img_dir /4T/CXY/Neural_Gaffer_original/objaverse_jobs/official_2000/raw",
        ),
    ),
    "ecommerce": DatasetSpec(
        status_json=ROOT / "subdataset" / "status_summary.json",
        raw_dir=ROOT / "external_sources" / "render_raw" / "ecommerce_subset_freeze",
        active_patterns=(
            r"run_render_completion_daemon\.sh ecommerce",
            r"run_subdataset_preprocess_job\.sh ecommerce",
            r"distribute-general-rendering\.py.*ecommerce",
            r"preprocess_rendered_image\.py --img_dir /4T/CXY/Neural_Gaffer_original/external_sources/render_raw/ecommerce_subset_freeze",
            r"preprocess_environment_map\.py --img_dir /4T/CXY/Neural_Gaffer_original/external_sources/render_raw/ecommerce_subset_freeze",
        ),
    ),
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def progress_counts(dataset: str, spec: DatasetSpec) -> tuple[int, int]:
    payload = load_payload(spec.status_json)
    if dataset == "official_2000":
        return int(payload.get("complete_train_object_count", 0)), int(payload.get("target_count", 0))
    theme = payload.get("themes", {}).get(dataset, {})
    return int(theme.get("complete_train_object_count", 0)), int(theme.get("target_count", 0))


def active_processes(spec: DatasetSpec) -> list[str]:
    matches: list[str] = []
    for pattern in spec.active_patterns:
        completed = subprocess.run(
            ["pgrep", "-af", pattern],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            for line in completed.stdout.splitlines():
                if line.strip():
                    matches.append(line.strip())
    deduped = sorted(set(matches))
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete dataset raw artifacts after processing finishes.")
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]

    while True:
        if not spec.status_json.exists():
            print(f"[{timestamp()}] dataset={args.dataset} waiting_for_status path={spec.status_json}", flush=True)
            time.sleep(args.interval_sec)
            continue

        complete, target = progress_counts(args.dataset, spec)
        active = active_processes(spec)
        print(
            f"[{timestamp()}] dataset={args.dataset} complete={complete} target={target} "
            f"active_processes={len(active)} raw_exists={spec.raw_dir.exists()}",
            flush=True,
        )

        if complete >= target > 0 and not active:
            if not spec.raw_dir.exists():
                print(f"[{timestamp()}] dataset={args.dataset} raw_missing path={spec.raw_dir}", flush=True)
                return 0
            if args.dry_run:
                print(f"[{timestamp()}] dataset={args.dataset} dry_run_delete path={spec.raw_dir}", flush=True)
                return 0
            print(f"[{timestamp()}] dataset={args.dataset} delete_start path={spec.raw_dir}", flush=True)
            shutil.rmtree(spec.raw_dir)
            print(f"[{timestamp()}] dataset={args.dataset} delete_done path={spec.raw_dir}", flush=True)
            return 0

        time.sleep(args.interval_sec)


if __name__ == "__main__":
    sys.exit(main())
