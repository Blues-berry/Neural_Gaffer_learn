# Highlight Evaluation Summary

- generated_at_utc: 2026-04-07T08:07:58.056509+00:00
- assets_manifest: /4T/CXY/Neural_Gaffer/effects/0407/highlight_test_samples_samebatch_v1/comparison/panel_analysis/ours_min_highlight_diff_page_01/selected_assets_manifest.json
- evaluated_methods: baseline, dilightnet, rgbx, ours

## baseline

| metric | mean |
| --- | ---: |
| full_psnr | 5.652463 |
| full_ssim | 0.241757 |
| foreground_psnr | 5.652463 |
| foreground_ssim | 0.241757 |
| highlight_psnr | 8.184213 |
| highlight_rmse | 0.397638 |
| highlight_mse_ratio | 0.585853 |
| highlight_mask_iou | 0.240415 |
| highlight_area_abs_error | 0.019180 |
| highlight_saturated_ratio_abs_error | 0.013589 |
| highlight_p95_luma_abs_error | 0.070156 |
| highlight_chroma_l1_on_gt_mask | 0.028854 |
| highlight_crop_ssim | 0.298581 |
| lpips_full | 0.620487 |
| lpips_foreground | 0.620487 |
| lpips_highlight_crop | 0.589488 |

| preset | full_psnr | full_ssim | foreground_psnr | highlight_psnr | highlight_mask_iou |
| --- | ---: | ---: | ---: | ---: | ---: |
| highlight_test_samebatch_onehdri | 5.652463 | 0.241757 | 5.652463 | 8.184213 | 0.240415 |

## dilightnet

| metric | mean |
| --- | ---: |
| full_psnr | 9.001495 |
| full_ssim | 0.731056 |
| foreground_psnr | 9.001495 |
| foreground_ssim | 0.731056 |
| highlight_psnr | 10.325225 |
| highlight_rmse | 0.315161 |
| highlight_mse_ratio | 0.902491 |
| highlight_mask_iou | 0.049401 |
| highlight_area_abs_error | 0.690668 |
| highlight_saturated_ratio_abs_error | 0.391494 |
| highlight_p95_luma_abs_error | 0.180448 |
| highlight_chroma_l1_on_gt_mask | 0.004427 |
| highlight_crop_ssim | 0.731056 |
| lpips_full | 0.395143 |
| lpips_foreground | 0.395143 |
| lpips_highlight_crop | 0.395143 |

| preset | full_psnr | full_ssim | foreground_psnr | highlight_psnr | highlight_mask_iou |
| --- | ---: | ---: | ---: | ---: | ---: |
| highlight_test_samebatch_onehdri | 9.001495 | 0.731056 | 9.001495 | 10.325225 | 0.049401 |

## rgbx

| metric | mean |
| --- | ---: |
| full_psnr | 8.808743 |
| full_ssim | 0.727359 |
| foreground_psnr | 8.808743 |
| foreground_ssim | 0.727359 |
| highlight_psnr | 9.806739 |
| highlight_rmse | 0.335467 |
| highlight_mse_ratio | 0.913525 |
| highlight_mask_iou | 0.051772 |
| highlight_area_abs_error | 0.693974 |
| highlight_saturated_ratio_abs_error | 0.391494 |
| highlight_p95_luma_abs_error | 0.180448 |
| highlight_chroma_l1_on_gt_mask | 0.002623 |
| highlight_crop_ssim | 0.727359 |
| lpips_full | 0.388892 |
| lpips_foreground | 0.388892 |
| lpips_highlight_crop | 0.388892 |

| preset | full_psnr | full_ssim | foreground_psnr | highlight_psnr | highlight_mask_iou |
| --- | ---: | ---: | ---: | ---: | ---: |
| highlight_test_samebatch_onehdri | 8.808743 | 0.727359 | 8.808743 | 9.806739 | 0.051772 |

## ours

| metric | mean |
| --- | ---: |
| full_psnr | 9.436906 |
| full_ssim | 0.741512 |
| foreground_psnr | 9.436906 |
| foreground_ssim | 0.741512 |
| highlight_psnr | 12.369852 |
| highlight_rmse | 0.249436 |
| highlight_mse_ratio | 0.569794 |
| highlight_mask_iou | 0.053098 |
| highlight_area_abs_error | 0.580467 |
| highlight_saturated_ratio_abs_error | 0.317815 |
| highlight_p95_luma_abs_error | 0.179259 |
| highlight_chroma_l1_on_gt_mask | 0.004364 |
| highlight_crop_ssim | 0.741512 |
| lpips_full | 0.401069 |
| lpips_foreground | 0.401069 |
| lpips_highlight_crop | 0.401069 |

| preset | full_psnr | full_ssim | foreground_psnr | highlight_psnr | highlight_mask_iou |
| --- | ---: | ---: | ---: | ---: | ---: |
| highlight_test_samebatch_onehdri | 9.436906 | 0.741512 | 9.436906 | 12.369852 | 0.053098 |

