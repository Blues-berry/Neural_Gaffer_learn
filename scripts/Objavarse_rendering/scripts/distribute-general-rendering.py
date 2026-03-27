import json
import multiprocessing
import shutil
import subprocess
import time
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import random
import boto3
import tyro
import wandb
import signal
import objaverse

@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""
    output_dir: str
    """output directory"""

    lighting_dir: str
    """output directory"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    objaverse_root: str = os.path.expanduser("~/.objaverse/hf-objaverse-v1")
    """Local Objaverse root. Relative paths from input_models_path are resolved under this root."""

    download_missing: bool = True
    """Download missing GLBs into objaverse_root before rendering."""

    seed: int = 42
    """Random seed used when shuffling objects."""

    max_objects: int = -1
    """Maximum number of objects to render. -1 means all."""

    min_free_gpu_mem_mb: int = 0
    """Wait until at least this much GPU memory is free before starting the next Blender render."""

    gpu_poll_interval_sec: int = 10
    """Polling interval while waiting for a GPU to have enough free memory."""

    download_mode: str = "direct"
    """How to download missing GLBs: direct, proxy, or curl-direct."""

    download_base_url: str = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
    """Base URL for downloading Objaverse GLBs. Can point to a mirror."""

    proxy_url: str = ""
    """Optional proxy URL, for example http://127.0.0.1:51081."""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""

    expected_views: int = 12
    """Expected number of camera views for a completed render."""

    expected_lighting_per_view: int = 16
    """Expected number of lighting conditions per view for a completed render."""


def load_model_map(input_models_path: str) -> Dict[str, str]:
    with open(input_models_path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        out = {}
        for item in payload:
            if isinstance(item, dict) and "uid" in item and "path" in item:
                out[item["uid"]] = item["path"]
            elif isinstance(item, str):
                uid = Path(item).stem
                out[uid] = item
            else:
                raise ValueError(f"Unsupported list entry in {input_models_path}: {item}")
        return out
    raise ValueError(f"Unsupported json payload in {input_models_path}: {type(payload)}")


def ensure_local_model(uid: str, relative_path: str, objaverse_root: str, download_missing: bool) -> str:
    local_path = os.path.join(objaverse_root, relative_path)
    if os.path.exists(local_path):
        return local_path
    if not download_missing:
        raise FileNotFoundError(f"Missing local GLB: {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_local_path = local_path + ".tmp"
    if os.path.exists(tmp_local_path):
        os.remove(tmp_local_path)
    download_url = f"{args.download_base_url.rstrip('/')}/{relative_path}"
    print(f"[download-start] uid={uid} mode={args.download_mode} proxy={args.proxy_url or 'none'} url={download_url}")
    download_file(download_url, tmp_local_path, mode=args.download_mode, proxy_url=args.proxy_url)
    os.rename(tmp_local_path, local_path)
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"[download-done] uid={uid} size_mb={size_mb:.2f} path={local_path}")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Download finished but file is still missing: {local_path}")
    return local_path


def download_file(url: str, destination: str, mode: str, proxy_url: str = "") -> None:
    if mode == "curl-direct":
        cmd = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--noproxy",
            "*",
            "--output",
            destination,
            url,
        ]
        subprocess.run(cmd, check=True)
        return
    if mode == "curl-proxy":
        cmd = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--proxy",
            proxy_url,
            "--output",
            destination,
            url,
        ]
        subprocess.run(cmd, check=True)
        return
    if mode == "proxy":
        import urllib.request
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
        with opener.open(url, timeout=120) as response, open(destination, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        return
    if mode == "direct":
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=120) as response, open(destination, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        return
    raise ValueError(f"Unsupported download_mode: {mode}")


def resolve_render_queue(model_map: Dict[str, str], max_objects: int, seed: int) -> List[Tuple[str, str]]:
    model_items = list(model_map.items())
    random.Random(seed).shuffle(model_items)
    if max_objects > 0:
        model_items = model_items[:max_objects]
    return model_items


def count_matches(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def summarize_render_output(path: Path, expected_views: int, expected_lighting_per_view: int) -> Dict[str, int]:
    expected_rgb = expected_views * expected_lighting_per_view
    return {
        "rgb_count": count_matches(path, "???_???_*.png"),
        "rt_count": count_matches(path, "*_RT.npy"),
        "normal_count": count_matches(path, "normal_*.png") + count_matches(path, "???_normals.png"),
        "random_lighting_count": count_matches(path, "random_lighting_*.png"),
        "expected_rgb": expected_rgb,
        "expected_views": expected_views,
    }


def render_output_is_complete(path: Path, expected_views: int, expected_lighting_per_view: int) -> bool:
    if not path.is_dir():
        return False
    summary = summarize_render_output(path, expected_views, expected_lighting_per_view)
    return (
        summary["rgb_count"] >= summary["expected_rgb"]
        and summary["rt_count"] >= summary["expected_views"]
        and summary["normal_count"] >= summary["expected_views"]
        and summary["random_lighting_count"] >= summary["expected_views"]
    )


def worker(queue: multiprocessing.JoinableQueue, count: multiprocessing.Value, gpu: int, s3: Optional[boto3.client]) -> None:
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break

        try:
            uid, object_path = item
            if not os.path.isabs(object_path):
                object_path = ensure_local_model(
                    uid,
                    object_path,
                    objaverse_root=args.objaverse_root,
                    download_missing=args.download_missing,
                )
            view_path = Path(OUT_DIR) / uid
            if render_output_is_complete(view_path, args.expected_views, args.expected_lighting_per_view):
                print("========", uid, "rendered", "========")
                with count.get_lock():
                    count.value += 1
                continue
            if view_path.exists():
                summary = summarize_render_output(view_path, args.expected_views, args.expected_lighting_per_view)
                print(f"[render-reset-incomplete] uid={uid} summary={summary}")
                if view_path.is_dir():
                    shutil.rmtree(view_path, ignore_errors=True)
                else:
                    view_path.unlink(missing_ok=True)

            lighting_dir = args.lighting_dir
            print(f"[render-start] uid={uid} gpu={gpu} object_path={object_path}")
            wait_for_gpu_headroom(gpu, args.min_free_gpu_mem_mb, args.gpu_poll_interval_sec)
            command = (
                f" blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
                f" --output_dir {shlex.quote(OUT_DIR)} "
                f" --object_path {shlex.quote(object_path)} "
                f" --object_uid {shlex.quote(uid)} "
                f" --test_light_dir {shlex.quote(lighting_dir)} "
            )

            result = subprocess.run(command, shell=True)
            print(f"[render-exit] uid={uid} gpu={gpu} returncode={result.returncode}")

            with count.get_lock():
                count.value += 1
        except Exception as exc:
            print(f"[worker-error] gpu={gpu} item={item} error={exc!r}")
        finally:
            queue.task_done()


def query_free_gpu_mem_mb(gpu: int) -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return 0
    physical_gpu = gpu
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        try:
            device_map = [int(item.strip()) for item in visible_devices.split(",") if item.strip()]
            if 0 <= gpu < len(device_map):
                physical_gpu = device_map[gpu]
        except Exception:
            physical_gpu = gpu
    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(physical_gpu),
            ],
            text=True,
        ).strip()
        return int(output.splitlines()[0])
    except Exception:
        return 0


def wait_for_gpu_headroom(gpu: int, min_free_gpu_mem_mb: int, gpu_poll_interval_sec: int) -> None:
    if min_free_gpu_mem_mb <= 0:
        return
    while True:
        free_mem_mb = query_free_gpu_mem_mb(gpu)
        if free_mem_mb >= min_free_gpu_mem_mb:
            return
        print(
            f"GPU {gpu} free memory {free_mem_mb} MB is below target {min_free_gpu_mem_mb} MB; waiting {gpu_poll_interval_sec}s before retry."
        )
        time.sleep(gpu_poll_interval_sec)


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    OUT_DIR = args.output_dir

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")
    processes = []
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(target=worker, args=(queue, count, gpu_i, s3))
            process.daemon = True
            process.start()
            processes.append(process)

    try:
        model_paths = load_model_map(args.input_models_path)
        render_items = resolve_render_queue(model_paths, max_objects=args.max_objects, seed=args.seed)

        for item in render_items:
            queue.put(item)

        if args.log_to_wandb:
            while True:
                time.sleep(5)
                wandb.log(
                    {
                        "count": count.value,
                        "total": len(render_items),
                        "progress": count.value / max(len(render_items), 1),
                    }
                )
                if count.value == len(render_items):
                    break

        queue.join()

        for i in range(args.num_gpus * args.workers_per_gpu):
            queue.put(None)
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Terminating processes.")
        for p in processes:
            os.kill(p.pid, signal.SIGKILL)
