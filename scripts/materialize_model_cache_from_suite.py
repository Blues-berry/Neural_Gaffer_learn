import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy suite model directories into a local cache root and rewrite the suite to cached paths."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-suite", required=True)
    parser.add_argument("--cache-root", default="/home/ubuntu/neural_gaffer_model_cache")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_value: str | None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_suite(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods", payload)
    if not methods:
        raise ValueError(f"No methods defined in {path}")
    return methods


def copy_tree(src: Path, dst: Path, force_refresh: bool):
    if dst.exists() and force_refresh:
        subprocess.run(["rm", "-rf", str(dst)], check=True)
    if (dst / "model_index.json").exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=True)


def main():
    args = parse_args()
    suite_path = resolve_repo_path(args.suite)
    output_suite_path = resolve_repo_path(args.output_suite)
    cache_root = resolve_repo_path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    cached_methods = []
    source_to_cached = {}
    for method in load_suite(suite_path):
        method_copy = dict(method)
        source_model_dir = resolve_repo_path(method_copy["model_dir"])
        if source_model_dir is None:
            raise ValueError(f"Method {method_copy.get('name')} has no model_dir")

        source_resolved = source_model_dir.resolve()
        cache_root_resolved = cache_root.resolve()
        if cache_root_resolved in source_resolved.parents or source_resolved == cache_root_resolved:
            cached_path = source_resolved
        else:
            cached_path = source_to_cached.get(str(source_resolved))
            if cached_path is None:
                target_name = source_model_dir.name
                cached_path = cache_root / target_name
                copy_tree(source_model_dir, cached_path, force_refresh=bool(args.force_refresh))
                source_to_cached[str(source_resolved)] = cached_path

        method_copy["model_dir"] = str(cached_path)
        checkpoint_path = method_copy.get("checkpoint_path")
        if checkpoint_path:
            checkpoint_resolved = resolve_repo_path(checkpoint_path)
            if checkpoint_resolved is not None:
                checkpoint_resolved = checkpoint_resolved.resolve()
                try:
                    relative_checkpoint = checkpoint_resolved.relative_to(source_resolved)
                    method_copy["checkpoint_path"] = str(cached_path / relative_checkpoint)
                except ValueError:
                    method_copy["checkpoint_path"] = str(checkpoint_resolved)
        cached_methods.append(method_copy)

    output_suite_path.parent.mkdir(parents=True, exist_ok=True)
    output_suite_path.write_text(json.dumps({"methods": cached_methods}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_suite_path}")


if __name__ == "__main__":
    main()
