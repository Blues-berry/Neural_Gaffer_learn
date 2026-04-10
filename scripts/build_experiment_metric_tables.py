import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_MAIN_METRICS = ["full_psnr", "foreground_psnr", "fg_rmse"]
DEFAULT_HIGHLIGHT_METRICS = [
    "highlight_psnr",
    "highlight_rmse",
    "highlight_mask_iou",
    "highlight_area_abs_error",
    "highlight_saturated_ratio_abs_error",
    "highlight_p95_luma_abs_error",
]
DEFAULT_LABELS = {
    "baseline": "Neural Gaffer",
    "dilightnet": "DiLightNet",
    "rgbx": "RGB<->X",
    "ours": "Ours",
    "ours_full": "Ours (Full)",
    "officialval": "Ours (OfficialVal)",
    "baseline_0316_fallback": "0316 Baseline",
    "jbhdfvfc_ckpt80k": "80K Highlight",
    "cosine0331_03": "Cosine 0331-03",
    "xkmlb19f_like_relative_fallback": "Relative Fallback",
    "hyblite_0331_02_fallback": "Abl. Hybrid Lite",
    "officialval_0403_04": "Ours (OfficialVal)",
    "abl00_base": "Abl. Base",
    "abl01_imgspace_fixed": "Abl. ImgSpace Fixed",
    "abl02_quantile": "Abl. Quantile",
    "abl03_blur": "Abl. Blur",
    "abl04_relative": "Abl. Relative",
    "abl05_full_main": "Abl. Full Main",
    "hyblite": "Abl. Hybrid Lite",
    "freqsplit": "Abl. Freq Split",
    "cosine_lowlr": "Abl. Cosine LowLR",
}
LOWER_IS_BETTER = {
    "fg_rmse",
    "highlight_rmse",
    "highlight_area_abs_error",
    "highlight_saturated_ratio_abs_error",
    "highlight_p95_luma_abs_error",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build markdown and CSV metric tables for a comparison or ablation experiment."
    )
    parser.add_argument("--metrics-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--main-metrics", nargs="*", default=None)
    parser.add_argument("--highlight-metrics", nargs="*", default=None)
    parser.add_argument("--title-prefix", default="Experiment")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pretty_method_name(method_name: str):
    return DEFAULT_LABELS.get(method_name, method_name)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def format_value(value):
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.4f}"


def method_sort_key(method_name: str, summary: dict, primary_metric: str):
    methods = summary.get("methods", {})
    value = safe_float(methods.get(method_name, {}).get(primary_metric))
    if value is None:
        return (999999999.0, method_name)
    if primary_metric in LOWER_IS_BETTER:
        return (value, method_name)
    return (-value, method_name)


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


def render_markdown_table(title: str, rows: list[dict], metric_names: list[str]):
    lines = [f"# {title}", ""]
    header = "| method | " + " | ".join(metric_names) + " |"
    divider = "| --- | " + " | ".join(["---:" for _ in metric_names]) + " |"
    lines.append(header)
    lines.append(divider)
    for row in rows:
        lines.append(
            "| "
            + row["label"]
            + " | "
            + " | ".join(format_value(row.get(metric_name)) for metric_name in metric_names)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_csv(rows: list[dict], metric_names: list[str]):
    lines = []
    fieldnames = ["method", "label", *metric_names]
    lines.append(",".join(fieldnames))
    for row in rows:
        values = [row["method"], row["label"], *[format_value(row.get(metric_name)) for metric_name in metric_names]]
        escaped = []
        for value in values:
            text = str(value)
            if "," in text or "\"" in text:
                text = "\"" + text.replace("\"", "\"\"") + "\""
            escaped.append(text)
        lines.append(",".join(escaped))
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    summary_path = resolve_repo_path(args.metrics_summary)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = load_json(summary_path)

    main_metrics = list(args.main_metrics or DEFAULT_MAIN_METRICS)
    highlight_metrics = list(args.highlight_metrics or DEFAULT_HIGHLIGHT_METRICS)
    methods = list(args.methods or summary.get("methods", {}).keys())
    methods = [method for method in methods if method in summary.get("methods", {})]

    if not methods:
        raise RuntimeError(f"No valid methods found in {summary_path}")

    methods.sort(key=lambda name: method_sort_key(name, summary, main_metrics[0]))

    main_rows = build_rows(summary, methods, main_metrics)
    highlight_rows = build_rows(summary, methods, highlight_metrics)

    global_md = render_markdown_table(f"{args.title_prefix} Main Metrics Table", main_rows, main_metrics)
    highlight_md = render_markdown_table(f"{args.title_prefix} Highlight Metrics Table", highlight_rows, highlight_metrics)

    (output_root / "global_quality_table.md").write_text(global_md, encoding="utf-8")
    (output_root / "global_quality_table.csv").write_text(render_csv(main_rows, main_metrics), encoding="utf-8")
    (output_root / "highlight_quality_table.md").write_text(highlight_md, encoding="utf-8")
    (output_root / "highlight_quality_table.csv").write_text(render_csv(highlight_rows, highlight_metrics), encoding="utf-8")

    (output_root / "tables_summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics_summary": str(summary_path),
                "output_root": str(output_root),
                "methods": methods,
                "main_metrics": main_metrics,
                "highlight_metrics": highlight_metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_root / 'global_quality_table.md'}")
    print(f"wrote {output_root / 'highlight_quality_table.md'}")


if __name__ == "__main__":
    main()
