import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MASK_FILENAME_PATTERNS = (
    "{view:03d}_mask.png",
    "{view:03d}_alpha.png",
    "mask_{view:03d}.png",
    "alpha_{view:03d}.png",
    "foreground_mask_{view:03d}.png",
    "foreground_alpha_{view:03d}.png",
    "random_lighting_{view:03d}.png",
)


def load_image_array(path: str):
    try:
        image = plt.imread(path)
    except Exception:
        return None
    return np.asarray(image)


def _to_mask_float(mask: np.ndarray):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)


def extract_alpha_or_mask(image: np.ndarray, source_path: str | None = None):
    if image is None:
        return None

    image = np.asarray(image)
    source_name = Path(source_path).name.lower() if source_path else ""

    if image.ndim == 3 and image.shape[-1] >= 4:
        return _to_mask_float(image[..., 3])

    if image.ndim == 2:
        return _to_mask_float(image)

    if image.ndim == 3 and image.shape[-1] >= 1 and any(token in source_name for token in ("mask", "alpha")):
        return _to_mask_float(image[..., 0])

    return None


def fallback_white_background_mask(rgb_image: np.ndarray, background_threshold: float = 0.98):
    if rgb_image is None:
        return None
    rgb_image = np.asarray(rgb_image, dtype=np.float32)
    if rgb_image.ndim == 2:
        rgb_image = np.repeat(rgb_image[..., None], 3, axis=-1)
    if rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[..., :3]
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    return (rgb_image.min(axis=-1) < float(background_threshold)).astype(np.float32)


def resolve_foreground_mask(object_dir: str, view_idx: int, reference_image_path: str | None = None):
    object_dir = Path(object_dir)

    candidate_paths = []
    if reference_image_path:
        candidate_paths.append(Path(reference_image_path))
    for pattern in MASK_FILENAME_PATTERNS:
        candidate_paths.append(object_dir / pattern.format(view=int(view_idx)))

    visited = set()
    for candidate_path in candidate_paths:
        candidate_path = Path(candidate_path)
        candidate_key = str(candidate_path.resolve()) if candidate_path.exists() else str(candidate_path)
        if candidate_key in visited or not candidate_path.exists():
            continue
        visited.add(candidate_key)

        image = load_image_array(str(candidate_path))
        mask = extract_alpha_or_mask(image, source_path=str(candidate_path))
        if mask is not None:
            return mask, str(candidate_path)

    return None, None
