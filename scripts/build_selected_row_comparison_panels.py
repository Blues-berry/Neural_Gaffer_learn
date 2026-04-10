import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select specific rows from existing all_methods page manifests and rebuild them into curated comparison panels."
    )
    parser.add_argument("--source-manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        help="Row spec in page:row format using 1-based indices, for example 01:4 or 17:4.",
    )
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--method-image-key", choices=["composited", "white_bg"], default="composited")
    parser.add_argument("--input-image-key", choices=["white", "composited"], default="white")
    parser.add_argument("--ground-truth-image-key", choices=["white", "composited"], default="composited")
    parser.add_argument("--visual-tag", default="input_white_methods_gt_hdrbg")
    return parser.parse_args()


def parse_row_spec(spec: str):
    page_text, row_text = spec.split(":", 1)
    return int(page_text), int(row_text)


def load_samples_for_spec(source_manifest_dir: Path, page_number: int, row_number: int):
    manifest_path = source_manifest_dir / f"all_methods_page_{page_number:02d}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    index = row_number - 1
    if index < 0 or index >= len(samples):
        raise IndexError(f"Row {row_number} is out of range for {manifest_path}")
    return samples[index], manifest_path


def write_manifest(samples: list[dict], output_path: Path):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_name": "requested_rows_curated",
        "visual_mode": "input_white_methods_gt_hdrbg",
        "sample_count": len(samples),
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_panel_variants(
    manifest_path: Path,
    output_base: Path,
    tile_size: int,
    method_image_key: str,
    input_image_key: str,
    ground_truth_image_key: str,
):
    script_path = REPO_ROOT / "scripts" / "build_relighting_comparison_panel.py"
    common = [
        sys.executable,
        str(script_path),
        "--assets-manifest",
        str(manifest_path),
        "--columns",
        "input_image",
        "method:baseline",
        "method:dilightnet",
        "method:rgbx",
        "method:ours",
        "ground_truth",
        "target_lighting",
        "--tile-size",
        str(tile_size),
        "--padding",
        "14",
        "--header-height",
        "60",
        "--method-image-key",
        method_image_key,
        "--input-image-key",
        input_image_key,
        "--ground-truth-image-key",
        ground_truth_image_key,
        "--hide-row-labels",
    ]
    headers_path = output_base.with_name(output_base.stem + "_headers.png")
    no_text_path = output_base.with_name(output_base.stem + "_no_text.png")
    legacy_path = output_base.with_suffix(".png")
    subprocess.run(common + ["--output", str(headers_path)], cwd=REPO_ROOT, check=True)
    subprocess.run(common + ["--no-text", "--output", str(no_text_path)], cwd=REPO_ROOT, check=True)
    subprocess.run(common + ["--output", str(legacy_path)], cwd=REPO_ROOT, check=True)
    return {
        "headers": str(headers_path),
        "no_text": str(no_text_path),
        "legacy": str(legacy_path),
    }


def main():
    args = parse_args()
    source_manifest_dir = Path(args.source_manifest_dir)
    output_root = Path(args.output_root)
    manifest_output_dir = output_root / "panel_manifests"
    panel_output_dir = output_root / "panels"
    rows_meta = []
    selected_samples = []

    for spec in args.row:
        page_number, row_number = parse_row_spec(spec)
        sample, source_manifest_path = load_samples_for_spec(source_manifest_dir, page_number, row_number)
        selected_samples.append(sample)
        rows_meta.append(
            {
                "spec": spec,
                "page_number": page_number,
                "row_number": row_number,
                "sample_key": sample.get("sample_key"),
                "source_manifest": str(source_manifest_path),
            }
        )

    page_outputs = []
    chunk_size = max(int(args.page_size), 1)
    for page_index, start in enumerate(range(0, len(selected_samples), chunk_size), start=1):
        page_samples = selected_samples[start:start + chunk_size]
        manifest_path = manifest_output_dir / f"selected_rows_page_{page_index:02d}.json"
        write_manifest(page_samples, manifest_path)
        output_base = panel_output_dir / f"selected_rows_page_{page_index:02d}"
        page_outputs.append(
            build_panel_variants(
                manifest_path=manifest_path,
                output_base=output_base,
                tile_size=int(args.tile_size),
                method_image_key=args.method_image_key,
                input_image_key=args.input_image_key,
                ground_truth_image_key=args.ground_truth_image_key,
            )
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_dir": str(source_manifest_dir),
        "output_root": str(output_root),
        "page_size": int(args.page_size),
        "tile_size": int(args.tile_size),
        "visual_tag": args.visual_tag,
        "selected_row_count": len(selected_samples),
        "selected_rows": rows_meta,
        "pages": page_outputs,
    }
    summary_path = output_root / "selected_rows_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
