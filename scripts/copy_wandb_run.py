#!/usr/bin/env python3
import argparse
import math
from typing import Any

import wandb


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def is_simple_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def clean_history_row(row: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    step = row.get("_step")
    if step is not None:
        try:
            step = int(step)
        except Exception:
            step = None

    cleaned = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if is_simple_value(value):
            cleaned[key] = value
    return step, cleaned


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if is_simple_value(value):
            cleaned[key] = value
    return cleaned


def clean_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        if is_simple_value(value):
            cleaned[key] = value
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a finished W&B run into another project for side-by-side comparison."
    )
    parser.add_argument("--entity", required=True)
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--dest-project", required=True)
    parser.add_argument("--name-suffix", default=" [copied baseline]")
    parser.add_argument("--history-page-size", type=int, default=1000)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="If > 0, use run.history(samples=...) instead of scan_history() for a faster sampled copy.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    source_path = f"{args.entity}/{args.source_project}/{args.source_run_id}"
    src = api.run(source_path)

    src_config = clean_config(dict(src.config))
    src_summary = clean_summary(dict(src.summary))
    src_tags = list(src.tags) + [
        "copied-run",
        f"source-project:{args.source_project}",
        f"source-run:{args.source_run_id}",
    ]
    src_name = f"{src.name}{args.name_suffix}"

    with wandb.init(
        entity=args.entity,
        project=args.dest_project,
        name=src_name,
        config={
            **src_config,
            "copied_from_run_path": source_path,
            "copied_from_project": args.source_project,
            "copied_from_run_id": args.source_run_id,
        },
        tags=src_tags,
        notes=f"Copied from {source_path} for cross-run comparison.",
        reinit=True,
    ) as run:
        rows_without_step: list[dict[str, Any]] = []
        merged_rows_by_step: dict[int, dict[str, Any]] = {}

        history_rows = (
            src.history(samples=args.samples, pandas=False)
            if args.samples > 0
            else src.scan_history(page_size=args.history_page_size)
        )

        for row in history_rows:
            step, cleaned = clean_history_row(row)
            if not cleaned:
                continue
            if step is None:
                rows_without_step.append(cleaned)
            else:
                merged_rows_by_step.setdefault(step, {}).update(cleaned)

        logged_rows = 0
        last_step = None

        for row in rows_without_step:
            run.log(row)
            logged_rows += 1

        for step in sorted(merged_rows_by_step):
            run.log(merged_rows_by_step[step], step=step)
            last_step = step
            logged_rows += 1

        for key, value in src_summary.items():
            run.summary[key] = value
        run.summary["copied_history_rows"] = logged_rows
        if last_step is not None:
            run.summary["copied_last_step"] = last_step

        print(f"Created destination run: {run.path}")
        print(f"Copied history rows: {logged_rows}")
        if last_step is not None:
            print(f"Last copied step: {last_step}")


if __name__ == "__main__":
    main()
