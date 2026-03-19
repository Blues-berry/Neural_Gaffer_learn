# Image-Space Highlight Training Update

## Date

2026-03-19

## Summary

This update changes the default highlight training strategy in `Neural_Gaffer`:

1. The previous default direction, highlight reweighting in latent/noise space, is no longer the default.
2. The new default direction is an auxiliary highlight-aware constraint in image space.
3. Standard diffusion loss remains the main optimization target.

This change was made because the latent/noise-space highlight weighting could improve bright specular regions while also making the overall image too dark, reducing mid-tone and shadow fidelity.

## Effective Defaults

The current effective defaults are:

- `use_highlight_weighted_loss = false`
- `use_image_space_highlight_loss = true`
- `image_space_constraint_weight = 0.1`
- `highlight_loss_weight = 2.0`
- `highlight_threshold = 0.8`
- `highlight_soft_weighting = true`
- `highlight_gamma = 2.0`

## Current Single-GPU Trial Configuration

For the current single-GPU experiment on `GPU1`, the active config file is:

- `configs/neural_gaffer_training_gpu1_highlight.txt`

The current batch size used for this run is:

- `training_batch_size = 6`

This was reduced from `16` after a previous out-of-memory failure on a 32 GB GPU when image-space constraint was enabled.

## Code Changes

### 1. Training objective migration

The training loop now uses:

- standard diffusion loss as the main loss
- optional latent/noise-space highlight reweighting, disabled by default
- optional image-space highlight auxiliary loss, enabled by default

The image-space branch works as follows:

1. reconstruct `pred_x0` from `model_pred`, `noisy_latents`, and `timesteps`
2. decode `pred_x0` through the VAE
3. compute a highlight-aware weighted MSE in image space
4. combine it with the diffusion loss using `image_space_constraint_weight`

The total loss is now:

`loss = diffusion_loss + image_space_constraint_weight * image_space_loss`

### 2. Logging changes

Training now logs:

- `loss_diffusion`
- `loss_image_space_constraint`
- `highlight_region_ratio`
- `highlight_mse`
- `non_highlight_mse`
- `highlight_mse_ratio`

When image-space constraint is enabled, the highlight metrics are reported for image-space error behavior as well, using the `image_space/` prefix.

Validation highlight metrics are now based on final predicted images rather than latent/noise-space error maps.

### 3. W&B project behavior

W&B initialization now prefers reusing the most recent local W&B project name before falling back to `tracker_project_name`.

## Why This Change Was Needed

The previous latent/noise-space highlight weighting was useful for emphasizing bright specular errors, but it could bias optimization too strongly toward highlights. In practice, this risked:

- darker global brightness
- weaker mid-tone reconstruction
- lower shadow/detail readability
- visually correct highlights but worse overall realism

The image-space constraint keeps the main diffusion objective intact while adding a more interpretable auxiliary penalty closer to the final rendered image.

## Predicted Outcomes

### Likely improvements

- less global darkening than the latent-weighted version
- better preservation of mid-tones and dark-region structure
- easier interpretation of highlight diagnostics
- more stable tuning because auxiliary loss acts on decoded image behavior

### Possible risks

- extra VAE decode increases memory usage
- extra image-space branch increases step time
- if the auxiliary weight is too large, decoded-image supervision may still over-regularize brightness
- if the auxiliary weight is too small, highlight behavior may remain under-constrained

## Recommended Monitoring

During early training, monitor:

- `loss_diffusion`
- `loss_image_space_constraint`
- `highlight_mse_ratio`
- `image_space/highlight_mse_ratio`
- generated validation images for global brightness and mid-tone readability

Recommended interpretation:

- if `loss_image_space_constraint` is large and images become dark, reduce `image_space_constraint_weight`
- if highlights are still weak or unstable, increase `image_space_constraint_weight` slightly
- if highlight regions are over-emphasized while the image looks dim, reduce `highlight_loss_weight`

## Suggested Tuning Plan

### If images still look too dark

Try, in order:

1. reduce `image_space_constraint_weight` from `0.1` to `0.05`
2. reduce `highlight_loss_weight` from `2.0` to `1.5`
3. raise `highlight_threshold` from `0.8` to `0.85`

### If highlights are too weak

Try, in order:

1. increase `image_space_constraint_weight` from `0.1` to `0.15`
2. increase `highlight_loss_weight` from `2.0` to `2.5`
3. keep validation images fixed and compare side by side before further changes

### If memory usage is still too high

Try, in order:

1. reduce `training_batch_size` from `6` to `4`
2. reduce `dataloader_num_workers` if host RAM pressure appears
3. disable image-space constraint only as a last fallback for debugging

## Current Recommendation

For the current run, the most reasonable starting point is:

- keep `training_batch_size = 6`
- keep `image_space_constraint_weight = 0.1`
- keep latent/noise-space weighting disabled
- observe the first validation outputs before changing threshold or highlight weight

If the next outputs still show global darkening, the first parameter to reduce should be:

- `image_space_constraint_weight`
