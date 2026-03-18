"""
Physical Constraints for Specular/Highlight Enhancement in Neural Gaffer

This module implements physical-based constraints to improve specular handling
without increasing network complexity at inference time.

Key constraints:
1. Albedo Consistency: Albedo (intrinsic color) should remain consistent under different lighting
2. Lambertian + Specular decomposition: I = kd * Ld * (n·l) + ks * Ls * (n·h)^n
3. Rendered image = Diffuse + Specular (physically plausible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class PhysicalConstraints(nn.Module):
    """
    Physical constraints for specular-aware relighting training.
    These constraints are only active during training and do not affect inference.
    """
    
    def __init__(self, 
                 albedo_consistency_weight: float = 0.1,
                 specular_awareness_weight: float = 0.05,
                 roughness_weight: float = 0.02):
        super().__init__()
        self.albedo_consistency_weight = albedo_consistency_weight
        self.specular_awareness_weight = specular_awareness_weight
        self.roughness_weight = roughness_weight
        
    def estimate_albedo(self, image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Estimate albedo from image using a simple max-lit approach.
        Albedo ≈ max(I) across color channels (simplified).
        This is a rough estimate useful for constraint regularization.
        """
        # Simple albedo estimation: use the brightest pixels as approximation
        # In reality, this should be combined with known lighting
        albedo = image.max(dim=1, keepdim=True)[0].clamp(min=eps)
        return albedo.repeat(1, 3, 1, 1) if albedo.shape[1] == 1 else albedo
    
    def albedo_consistency_loss(self, 
                                pred_images: torch.Tensor, 
                                target_images: torch.Tensor,
                                input_images: torch.Tensor) -> torch.Tensor:
        """
        Albedo consistency constraint: The predicted image and target image
        should have similar albedo estimates (intrinsic color).
        """
        # Estimate albedo from predicted and target images
        pred_albedo = self.estimate_albedo(pred_images)
        target_albedo = self.estimate_albedo(target_images)
        
        # Albedo should be similar (especially for diffuse-dominant regions)
        # We use L1 loss for stability
        albedo_loss = F.l1_loss(pred_albedo, target_albedo)
        
        # Also constrain that albedo should be similar to input image's albedo
        input_albedo = self.estimate_albedo(input_images)
        input_albedo_loss = F.l1_loss(pred_albedo, input_albedo.detach())
        
        return albedo_loss + 0.5 * input_albedo_loss
    
    def diffuse_specular_decomposition(self, 
                                        image: torch.Tensor,
                                        threshold: float = 0.7) -> tuple:
        """
        Simple diffuse/specular decomposition based on intensity threshold.
        High intensity regions (> threshold percentile) are considered specular.
        """
        # Compute luminance
        luminance = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # Threshold for specular detection (highlights are typically bright)
        threshold_value = torch.quantile(luminance, threshold, dim=[2, 3], keepdim=True)
        
        # Create masks
        specular_mask = (luminance > threshold_value).float()
        diffuse_mask = 1.0 - specular_mask
        
        # Decompose
        specular = image * specular_mask
        diffuse = image * diffuse_mask
        
        return diffuse, specular, specular_mask
    
    def specular_awareness_loss(self, 
                                pred_images: torch.Tensor, 
                                target_images: torch.Tensor) -> torch.Tensor:
        """
        Specular awareness loss: Ensure high-intensity regions (highlights)
        are handled correctly.
        """
        # Decompose both predictions and targets
        pred_diffuse, pred_specular, _ = self.diffuse_specular_decomposition(pred_images)
        target_diffuse, target_specular, _ = self.diffuse_specular_decomposition(target_images)
        
        # Separate losses for diffuse and specular regions
        diffuse_loss = F.l1_loss(pred_diffuse, target_diffuse)
        specular_loss = F.l1_loss(pred_specular, target_specular)
        
        # Specular is harder to predict, give it slightly more weight
        return diffuse_loss + 1.5 * specular_loss
    
    def smoothness_prior(self, image: torch.Tensor) -> torch.Tensor:
        """
        Smoothness prior: Natural images should have smooth albedo.
        This helps reduce noise in high-frequency regions.
        """
        # Compute gradient-based smoothness loss
        dx = image[:, :, :, 1:] - image[:, :, :, :-1]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        smoothness = dx.abs().mean() + dy.abs().mean()
        return smoothness
    
    def roughness_weighted_loss(self,
                                pred_images: torch.Tensor,
                                target_images: torch.Tensor) -> torch.Tensor:
        """
        Material roughness-aware loss: Different materials have different
        roughness characteristics. This encourages the model to learn
        material-appropriate specular behavior.
        """
        # Compute local variance as roughness proxy
        def compute_roughness(img):
            # Use a small window to compute local variance
            B, C, H, W = img.shape
            # Flatten spatial dims
            img_flat = img.view(B, C, -1)
            mean = img_flat.mean(dim=2, keepdim=True)
            var = ((img_flat - mean) ** 2).mean(dim=2, keepdim=True)
            return var.view(B, C, 1, 1)
        
        pred_roughness = compute_roughness(pred_images)
        target_roughness = compute_roughness(target_images)
        
        # Loss encourages correct roughness handling
        return F.l1_loss(pred_roughness, target_roughness)
    
    def forward(self, 
                pred_images: torch.Tensor, 
                target_images: torch.Tensor,
                input_images: torch.Tensor) -> dict:
        """
        Compute all physical constraints.
        
        Args:
            pred_images: Predicted relit images [B, 3, H, W]
            target_images: Ground truth relit images [B, 3, H, W]
            input_images: Input condition images [B, 3, H, W]
            
        Returns:
            Dictionary of losses
        """
        # Ensure images are in [0, 1] range for albedo estimation
        # (assuming input is in [-1, 1], convert)
        pred_norm = (pred_images + 1) / 2
        target_norm = (target_images + 1) / 2
        input_norm = (input_images + 1) / 2
        
        losses = {}
        
        # Albedo consistency loss
        if self.albedo_consistency_weight > 0:
            losses['albedo_consistency'] = self.albedo_consistency_loss(
                pred_norm, target_norm, input_norm
            ) * self.albedo_consistency_weight
            
        # Specular awareness loss
        if self.specular_awareness_weight > 0:
            losses['specular_awareness'] = self.specular_awareness_loss(
                pred_norm, target_norm
            ) * self.specular_awareness_weight
            
        # Roughness-weighted loss
        if self.roughness_weight > 0:
            losses['roughness'] = self.roughness_weighted_loss(
                pred_norm, target_norm
            ) * self.roughness_weight
            
        # Smoothness prior (optional, helps with albedo estimation)
        albedo_est = self.estimate_albedo(pred_norm)
        losses['smoothness'] = self.smoothness_prior(albedo_est) * 0.01
        
        return losses


class HighlightAwareLoss(nn.Module):
    """
    A simplified highlight-aware loss that can be added to the training loop.
    This is a drop-in replacement for the standard MSE loss with additional
    highlight awareness.
    """
    
    def __init__(self, 
                 base_weight: float = 1.0,
                 highlight_weight: float = 0.2,
                 highlight_threshold: float = 0.8):
        super().__init__()
        self.base_weight = base_weight
        self.highlight_weight = highlight_weight
        self.highlight_threshold = highlight_threshold
        self.physical_constraints = PhysicalConstraints()
        
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                input_image: torch.Tensor = None) -> torch.Tensor:
        """
        Compute highlight-aware loss.
        
        Args:
            pred: Predicted noise or image [B, C, H, W]
            target: Target noise or image [B, C, H, W]
            input_image: Optional input condition image
        """
        # Base MSE loss
        base_loss = F.mse_loss(pred, target)
        
        total_loss = base_loss * self.base_weight
        
        # If input_image is provided, add physical constraints
        if input_image is not None:
            # Decode latents to images if needed (assumes pred/target are latents)
            # This is a simplified version; in practice, you might want to
            # add this loss at the image level after VAE decode
            
            # For now, we apply a simpler approach: weighted loss on bright regions
            target_norm = (target + 1) / 2  # Convert to [0, 1]
            luminance = 0.299 * target_norm[:, 0:1] + 0.587 * target_norm[:, 1:2] + 0.114 * target_norm[:, 2:3]
            
            # Highlight regions (bright pixels)
            highlight_mask = (luminance > self.highlight_threshold).float()
            
            # Weighted loss: emphasize highlights
            if highlight_mask.sum() > 0:
                # Compute MSE only on highlights
                highlight_loss = ((pred - target) ** 2 * highlight_mask).sum() / (highlight_mask.sum() + 1e-6)
                total_loss = total_loss + highlight_loss * self.highlight_weight
                
        return total_loss


def apply_physical_constraints(pred_images: torch.Tensor,
                                target_images: torch.Tensor,
                                input_images: torch.Tensor,
                                weights: dict = None) -> dict:
    """
    Convenience function to apply physical constraints.
    
    Args:
        pred_images: Predicted images in [-1, 1] range
        target_images: Target images in [-1, 1] range  
        input_images: Input condition images in [-1, 1] range
        weights: Dictionary of loss weights
        
    Returns:
        Dictionary of losses
    """
    if weights is None:
        weights = {
            'albedo_consistency': 0.1,
            'specular_awareness': 0.05,
            'roughness': 0.02
        }
    
    constraints = PhysicalConstraints(
        albedo_consistency_weight=weights.get('albedo_consistency', 0.1),
        specular_awareness_weight=weights.get('specular_awareness', 0.05),
        roughness_weight=weights.get('roughness', 0.02)
    )
    
    return constraints(pred_images, target_images, input_images)


# Utility functions for data augmentation with highlight awareness
def highlight_aware_augmentation(image: torch.Tensor, 
                                  envmap: torch.Tensor,
                                  highlight_prob: float = 0.3) -> tuple:
    """
    Apply highlight-aware data augmentation during training.
    This can help the model learn better specular handling.
    
    Args:
        image: Input image [3, H, W]
        envmap: Environment map [3, H, W]
        highlight_prob: Probability of applying highlight augmentation
        
    Returns:
        Augmented (image, envmap)
    """
    if torch.rand(1).item() > highlight_prob:
        return image, envmap
    
    # Randomly adjust brightness of environment map to simulate
    # different specular intensity conditions
    brightness_factor = torch.rand(1).item() * 0.5 + 0.5  # [0.5, 1.0]
    envmap_aug = envmap * brightness_factor
    
    return image, envmap_aug


def compute_specular_metrics(pred: torch.Tensor, 
                              target: torch.Tensor) -> dict:
    """
    Compute metrics specifically for specular/highlight quality.
    
    Returns:
        Dictionary of specular metrics
    """
    # Normalize to [0, 1]
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    
    # Luminance
    pred_lum = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
    target_lum = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
    
    # High-light regions (top 20%)
    threshold = 0.8
    pred_highlight = pred_lum > threshold
    target_highlight = target_lum > threshold
    
    # Compute metrics
    metrics = {}
    
    # Highlight region accuracy
    if pred_highlight.sum() > 0 and target_highlight.sum() > 0:
        # Intersection over Union for highlight regions
        intersection = (pred_highlight & target_highlight).float().sum()
        union = (pred_highlight | target_highlight).float().sum()
        metrics['highlight_iou'] = (intersection / (union + 1e-6)).item()
        
        # Mean intensity in highlight regions
        pred_highlight_intensity = pred_lum[pred_highlight].mean()
        target_highlight_intensity = target_lum[target_highlight].mean()
        metrics['highlight_intensity_error'] = abs(
            pred_highlight_intensity - target_highlight_intensity
        ).item()
    
    return metrics
