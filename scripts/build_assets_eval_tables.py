import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from build_experiment_metric_tables import (
    DEFAULT_HIGHLIGHT_METRICS,
    DEFAULT_LABELS,
    DEFAULT_MAIN_METRICS,
    format_value,
    pretty_method_name,
    render_csv,
    render_markdown_table,
    resolve_repo_path,
    safe_float,
)


LPIPS_METRICS = ["lpips_full", "lpips_foreground", "lpips_highlight_crop"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build comparison-style tables from nested assets-level evaluation JSON."
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--title-prefix", default="Experiment")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_method_overall(overall: dict):
    flat = {}
    for metric_name, payload in overall.items():
        if isinstance(payload, dict):
            mean_value = payload.get("mean")
            if mean_value is not None:
                flat[metric_name] = mean_value
        elif payload is not None:
            flat[metric_name] = payload

    foreground_mse = safe_float(flat.get("foreground_mse"))
    if foreground_mse is not None:
        flat["fg_rmse"] = foreground_mse**0.5
    return flat


def flatten_summary(eval_payload: dict, methods: list[str]):
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_eval_json": eval_payload.get("source_eval_json"),
        "methods": {
            method_name: flatten_method_overall(eval_payload["methods"][method_name].get("overall", {}))
            for method_name in methods
        },
    }


def build_rows(summary: dict, method_order: list[str], metric_names: list[str]):
    methods = summary.get("methods", {})
    rows = []
    for method_name in method_order:
        payload = methods.get(method_name, {})
        row = {
            "method": method_name,
            "label": pretty_method_name(method_name),
        }
        for metric_name in metric_names:
            row[metric_name] = safe_float(payload.get(metric_name))
        rows.append(row)
    return rows


def sort_methods(summary: dict, methods: list[str], metric_name: str, lower_is_better: bool):
    def key(method_name: str):
        value = safe_float(summary.get("methods", {}).get(method_name, {}).get(metric_name))
        if value is None:
            return (999999999.0, method_name)
        return (value, method_name) if lower_is_better else (-value, method_name)

    return sorted(methods, key=key)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_per_sample_csv(eval_payload: dict, methods: list[str], output_path: Path):
    rows = []
    metric_names = set()
    for method_name in methods:
        for sample in eval_payload["methods"][method_name].get("samples", []):
            metrics = dict(sample.get("metrics", {}))
            foreground_mse = safe_float(metrics.get("foreground_mse"))
            if foreground_mse is not None:
                metrics["fg_rmse"] = foreground_mse**0.5
            metric_names.update(metrics.keys())
            rows.append(
                {
                    "method": method_name,
                    "label": pretty_method_name(method_name),
                    "sample_key": sample.get("sample_key"),
                    "preset": sample.get("preset"),
                    "object_id": sample.get("object_id"),
                    "view_idx": sample.get("view_idx"),
                    "target_file": sample.get("target_file"),
                    **metrics,
                }
            )

    fieldnames = [
        "method",
        "label",
        "sample_key",
        "preset",
        "object_id",
        "view_idx",
        "target_file",
        *sorted(metric_names),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_readme(
    output_root: Path,
    eval_json_path: Path,
    methods: list[str],
):
    lines = [
        "# Foreground Highlight Supervision Ablation (Comparison-Style Eval)",
        "",
        f"- source eval json: `{eval_json_path}`",
        f"- generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- methods: {', '.join(DEFAULT_LABELS.get(name, name) for name in methods)}",
        "",
        "## Files",
        "",
        f"- [metrics_summary_flat.json]({output_root / 'metrics_summary_flat.json'})",
        f"- [highlight_metrics_per_sample.csv]({output_root / 'highlight_metrics_per_sample.csv'})",
        f"- [global_quality_table.md]({output_root / 'tables' / 'global_quality_table.md'})",
        f"- [highlight_quality_table.md]({output_root / 'tables' / 'highlight_quality_table.md'})",
        f"- [lpips_quality_table.md]({output_root / 'tables' / 'lpips_quality_table.md'})",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    eval_json_path = resolve_repo_path(args.eval_json)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    eval_payload = load_json(eval_json_path)
    eval_payload["source_eval_json"] = str(eval_json_path)
    available_methods = list(eval_payload.get("methods", {}).keys())
    methods = [name for name in (args.methods or available_methods) if name in eval_payload.get("methods", {})]
    if not methods:
        raise RuntimeError(f"No valid methods found in {eval_json_path}")

    summary = flatten_summary(eval_payload, methods)
    metrics_summary_path = output_root / "metrics_summary_flat.json"
    write_text(metrics_summary_path, json.dumps(summary, indent=2) + "\n")

    tables_root = output_root / "tables"
    main_methods = sort_methods(summary, methods, DEFAULT_MAIN_METRICS[0], lower_is_better=False)
    highlight_methods = sort_methods(summary, methods, DEFAULT_HIGHLIGHT_METRICS[0], lower_is_better=False)
    lpips_methods = sort_methods(summary, methods, LPIPS_METRICS[0], lower_is_better=True)

    main_rows = build_rows(summary, main_methods, DEFAULT_MAIN_METRICS)
    highlight_rows = build_rows(summary, highlight_methods, DEFAULT_HIGHLIGHT_METRICS)
    lpips_rows = build_rows(summary, lpips_methods, LPIPS_METRICS)

    write_text(
        tables_root / "global_quality_table.md",
        render_markdown_table(f"{args.title_prefix} Main Metrics Table", main_rows, DEFAULT_MAIN_METRICS),
    )
    write_text(tables_root / "global_quality_table.csv", render_csv(main_rows, DEFAULT_MAIN_METRICS))
    write_text(
        tables_root / "highlight_quality_table.md",
        render_markdown_table(f"{args.title_prefix} Highlight Metrics Table", highlight_rows, DEFAULT_HIGHLIGHT_METRICS),
    )
    write_text(tables_root / "highlight_quality_table.csv", render_csv(highlight_rows, DEFAULT_HIGHLIGHT_METRICS))
    write_text(
        tables_root / "lpips_quality_table.md",
        render_markdown_table(f"{args.title_prefix} LPIPS Table", lpips_rows, LPIPS_METRICS),
    )
    write_text(tables_root / "lpips_quality_table.csv", render_csv(lpips_rows, LPIPS_METRICS))
    write_text(
        tables_root / "tables_summary.json",
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "eval_json": str(eval_json_path),
                "metrics_summary_flat": str(metrics_summary_path),
                "methods": methods,
                "main_metrics": DEFAULT_MAIN_METRICS,
                "highlight_metrics": DEFAULT_HIGHLIGHT_METRICS,
                "lpips_metrics": LPIPS_METRICS,
            },
            indent=2,
        )
        + "\n",
    )

    build_per_sample_csv(eval_payload, methods, output_root / "highlight_metrics_per_sample.csv")
    write_text(output_root / "README.md", build_readme(output_root, eval_json_path, methods))
    print(f"wrote {metrics_summary_path}")
    print(f"wrote {tables_root / 'global_quality_table.md'}")
    print(f"wrote {tables_root / 'highlight_quality_table.md'}")
    print(f"wrote {tables_root / 'lpips_quality_table.md'}")


if __name__ == "__main__":
    main()
