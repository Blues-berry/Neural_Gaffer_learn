from dataclasses import dataclass

import torch
import torch.nn as nn
import kornia


def _resolve_blur_kernel_size(kernel_size: int, sigma: float) -> int:
    kernel_size = int(kernel_size or 0)
    sigma = float(sigma or 0.0)
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size
    if sigma <= 0.0:
        return 1
    radius = max(int(round(3.0 * sigma)), 1)
    return radius * 2 + 1


def _per_pixel_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    loss_type = str(loss_type or "l1").lower()
    diff = prediction - target
    if loss_type == "l1":
        return diff.abs()
    if loss_type == "l2":
        return diff.pow(2)
    raise ValueError(f"Unsupported frequency loss type: {loss_type}")


def _weighted_channel_mean(
    per_pixel_loss: torch.Tensor,
    weight_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    weight_map = weight_map.float()
    if weight_map.ndim != per_pixel_loss.ndim:
        raise ValueError(
            f"weight_map ndim {weight_map.ndim} must match per_pixel_loss ndim {per_pixel_loss.ndim}"
        )
    if weight_map.shape[1] == 1 and per_pixel_loss.shape[1] != 1:
        norm = weight_map.sum() * per_pixel_loss.shape[1]
    else:
        norm = weight_map.sum()
    return (per_pixel_loss.float() * weight_map).sum() / norm.clamp_min(float(eps))


@dataclass
class FrequencySeparationAuxiliaryOutput:
    total_loss: torch.Tensor
    low_frequency_loss: torch.Tensor
    high_frequency_loss: torch.Tensor
    pred_low_abs_mean: torch.Tensor
    gt_low_abs_mean: torch.Tensor
    pred_high_abs_mean: torch.Tensor
    gt_high_abs_mean: torch.Tensor


class FrequencySeparationAuxiliaryLoss(nn.Module):
    """
    Standalone low/high-frequency auxiliary loss block.

    Design goals:
    - Keep the existing highlight-definition pipeline untouched.
    - Separate smooth illumination/soft-shadow consistency from sharp highlight/detail consistency.
    - Accept externally computed foreground masks and highlight-weight maps so ablations stay clean.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_image: torch.Tensor,
        gt_image: torch.Tensor,
        *,
        foreground_mask: torch.Tensor | None,
        highlight_weight_map: torch.Tensor | None,
        blur_sigma: float,
        blur_kernel_size: int = 0,
        low_loss_weight: float = 0.0,
        high_loss_weight: float = 0.0,
        low_loss_type: str = "l1",
        high_loss_type: str = "l1",
    ) -> FrequencySeparationAuxiliaryOutput:
        pred_image = pred_image.float()
        gt_image = gt_image.float()

        if foreground_mask is None:
            foreground_mask = torch.ones_like(pred_image[:, :1], dtype=pred_image.dtype, device=pred_image.device)
        else:
            foreground_mask = foreground_mask.to(device=pred_image.device, dtype=pred_image.dtype)

        if highlight_weight_map is None:
            highlight_weight_map = foreground_mask
        else:
            highlight_weight_map = highlight_weight_map.to(device=pred_image.device, dtype=pred_image.dtype)

        kernel_size = _resolve_blur_kernel_size(blur_kernel_size, blur_sigma)
        sigma = float(blur_sigma or 0.0)
        if kernel_size <= 1 or sigma <= 0.0:
            pred_low = pred_image
            gt_low = gt_image
        else:
            kernel = (kernel_size, kernel_size)
            sigma_pair = (sigma, sigma)
            pred_low = kornia.filters.gaussian_blur2d(pred_image, kernel_size=kernel, sigma=sigma_pair)
            gt_low = kornia.filters.gaussian_blur2d(gt_image, kernel_size=kernel, sigma=sigma_pair)

        pred_high = pred_image - pred_low
        gt_high = gt_image - gt_low

        low_per_pixel_loss = _per_pixel_loss(pred_low, gt_low, low_loss_type)
        high_per_pixel_loss = _per_pixel_loss(pred_high, gt_high, high_loss_type)

        low_frequency_loss = _weighted_channel_mean(low_per_pixel_loss, foreground_mask)
        high_frequency_loss = _weighted_channel_mean(high_per_pixel_loss, highlight_weight_map)

        total_loss = (
            float(low_loss_weight) * low_frequency_loss
            + float(high_loss_weight) * high_frequency_loss
        )

        return FrequencySeparationAuxiliaryOutput(
            total_loss=total_loss,
            low_frequency_loss=low_frequency_loss,
            high_frequency_loss=high_frequency_loss,
            pred_low_abs_mean=pred_low.abs().mean(),
            gt_low_abs_mean=gt_low.abs().mean(),
            pred_high_abs_mean=pred_high.abs().mean(),
            gt_high_abs_mean=gt_high.abs().mean(),
        )
