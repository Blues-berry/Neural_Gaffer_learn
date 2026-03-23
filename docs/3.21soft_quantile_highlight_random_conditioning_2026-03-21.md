# Soft-Quantile Highlight Random Conditioning (SQH-RC)

Date: 2026-03-21

## Method Definition

SQH-RC refers to the current training recipe used in W&B run `nsniycxi`:

- image-space highlight loss enabled
- soft highlight weighting enabled
- quantile-based highlight threshold enabled
- `highlight_quantile = 0.9`
- `highlight_min_threshold = 0.6`
- `highlight_max_threshold = 0.9`
- `random_lighting_condition_prob = 0.4`
- short training horizon: `18000` steps

This method is intended to improve robustness to broad, soft highlights from area-light conditions while keeping training shorter and more stable than the earlier `80000`-step runs.

## Compared Runs

| Label | W&B Run | Key Setting Summary | Steps |
| --- | --- | --- | --- |
| Baseline | `5y90hbdc` | soft highlight weighting, no image-space highlight loss | 80000 |
| Fixed-threshold image-space highlight | `qju56ygl` | image-space highlight loss, fixed threshold `0.8` | 80000 |
| SQH-RC | `nsniycxi` | image-space highlight loss + quantile mask + random-light conditioning | 18000 |

## Quantitative Comparison

| Metric | Baseline `5y90hbdc` | Fixed-threshold `qju56ygl` | SQH-RC `nsniycxi` | SQH-RC vs Baseline | SQH-RC vs Fixed-threshold |
| --- | --- | --- | --- | --- | --- |
| `PSNR/train` | 28.4391 | 30.0214 | 27.6715 | -0.7676 | -2.3500 |
| `PSNR/training_object_with_unseen_envir` | 31.0940 | 30.0620 | 29.6327 | -1.4613 | -0.4293 |
| `PSNR/unseen_object_with_seen_envir` | 29.5361 | 29.3631 | 29.5849 | +0.0488 | +0.2218 |
| `PSNR/unseen_object_with_unseen_envir` | 30.5093 | 30.0675 | 30.7810 | +0.2717 | +0.7135 |
| `PSNR/unseen_object_with_random_area_light_condition` | 28.0132 | 26.5557 | 25.4691 | -2.5442 | -1.0866 |

## Main Findings

### 1. The current method improves the main unseen-environment target

The clearest win is `PSNR/unseen_object_with_unseen_envir = 30.7810`, which is:

- `+0.2717 dB` over the baseline
- `+0.7135 dB` over the previous fixed-threshold image-space highlight run

This suggests the quantile-based soft highlight mask is helping the model generalize better to the main relighting target under unseen objects and unseen environments.

### 2. The current method is slightly better on `unseen_object_with_seen_envir`

`PSNR/unseen_object_with_seen_envir = 29.5849` is only a small gain, but it is still:

- `+0.0488 dB` over the baseline
- `+0.2218 dB` over the fixed-threshold image-space highlight run

This means the new method is not only helping the most challenging unseen-environment split; it also improves the seen-environment generalization split slightly.

### 3. The random area-light split is still the main weakness

`PSNR/unseen_object_with_random_area_light_condition = 25.4691` remains the worst metric:

- `-2.5442 dB` compared with the baseline
- `-1.0866 dB` compared with the fixed-threshold image-space highlight run

So the new method did not recover the robustness gap on the random area-light condition split.

### 4. The shorter training recipe is stable and avoids the late-stage collapse

Unlike the earlier long run that collapsed after `74k`, the current run completed to `18000` steps, kept normal prediction brightness, and produced valid final metrics.

This makes the short-horizon recipe a better default training schedule for this lightweight dataset even though it is not yet the best configuration for the random area-light split.

## Rendering-Oriented Interpretation

The current run shows a split-specific pattern:

- on `unseen_object_with_unseen_envir`, the prediction brightness is close to the target brightness
- on `unseen_object_with_random_area_light_condition`, the prediction remains slightly biased toward the condition image brightness

From the final brightness summary of `nsniycxi`:

- `unseen_object_with_random_area_light_condition`
  - `pred_mean - gt_mean = +0.015231`
  - `input_mean - gt_mean = +0.025878`
  - `pred_mean - input_mean = -0.010647`
- `unseen_object_with_unseen_envir`
  - `pred_mean - gt_mean = -0.004148`
  - `input_mean - gt_mean = +0.000916`
  - `pred_mean - input_mean = -0.005064`
- `unseen_object_with_seen_envir`
  - `pred_mean - gt_mean = -0.002396`
  - `input_mean - gt_mean = -0.007578`
  - `pred_mean - input_mean = +0.005183`

This supports the earlier diagnosis:

- the model is doing a better job matching the target environment under normal unseen-environment relighting
- under random area-light condition images, it still preserves too much of the input-lighting appearance

In other words, SQH-RC improves main-task relighting generalization, but condition-image OOD robustness is still lagging.

## Why SQH-RC Helps

The likely reasons the current method improves the main target split are:

- the quantile threshold is less brittle than a fixed `0.8` highlight cutoff
- the soft weighting reduces the risk of noisy hard-mask boundaries
- `random_lighting_condition_prob = 0.4` exposes the model to more lighting variation during training
- the `18000`-step budget avoids unnecessary late training where the earlier setup became unstable

## Remaining Bottlenecks

### 1. The highlight mask is still too sparse for area-light highlights

Even in the current run, the final logged highlight statistics remain very sparse:

- `panel_brightness/unseen_object_with_random_area_light_condition/highlight_mask/mean = 0.006124`
- `panel_brightness/unseen_object_with_random_area_light_condition/highlight_weight/mean = 0.002393`

This suggests the current quantile mask is more stable than a fixed threshold, but it may still be too selective for broad specular lobes created by area lights.

### 2. The model still over-trusts the condition image under area-light OOD

The random area-light split differs from the unseen-environment split mainly because the condition image itself is OOD. The current brightness gap indicates the prediction is still partially pulled toward the input appearance instead of fully matching the target lighting.

### 3. Short training may also reduce in-domain fitting

The drop on `PSNR/train` and `PSNR/training_object_with_unseen_envir` suggests that `18000` steps plus stronger augmentation is trading some in-domain fit for better main-task generalization.

That tradeoff is acceptable if the priority is unseen-environment performance, but it should be monitored if training-set fidelity matters.

## Improvement Space

### Priority 1: make the area-light highlight region less sparse

Recommended next ablations:

- reduce `highlight_quantile` from `0.90` to `0.85` or `0.88`
- keep the soft weighting, but widen the effective highlight region
- optionally smooth the highlight map before thresholding so wide area-light lobes are not fragmented

Expected benefit:

- better coverage of broad specular responses
- less under-detection of valid area-light highlight regions

### Priority 2: push condition-image robustness a bit further

Recommended next ablations:

- raise `random_lighting_condition_prob` from `0.4` to `0.5`
- or use a curriculum such as `0.3 -> 0.5` instead of jumping immediately to a very high ratio

Expected benefit:

- less reliance on the condition-image lighting prior
- better transfer to `unseen_object_with_random_area_light_condition`

### Priority 3: rebalance the loss only for area-light-like samples

If the random area-light split remains weak, the next step should not be increasing all highlight pressure globally. A better direction is:

- upweight image-space highlight supervision only for random-light-conditioned samples
- or apply a slightly lower quantile threshold only on those samples

Expected benefit:

- preserve the gain on `unseen_object_with_unseen_envir`
- avoid hurting the standard relighting distribution while specifically targeting the weak split

### Priority 4: keep the short schedule

For this dataset, `15000-20000` steps is already enough to reach convergence. The current result supports using a short schedule as the default.

Recommended default:

- keep `18000` steps
- save every `1000` steps
- continue selecting the best checkpoint within the training window

## Recommended Next Experiment

The most targeted follow-up is:

1. keep the current SQH-RC recipe
2. set `highlight_quantile = 0.85` or `0.88`
3. keep `random_lighting_condition_prob = 0.4` for the first retry
4. run again for `18000` steps

If that improves the random area-light split without sacrificing `unseen_object_with_unseen_envir`, then the next step should be testing `random_lighting_condition_prob = 0.5`.

## Reference Files

Metrics and configs:

- `wandb/run-20260318_104331-5y90hbdc/files/wandb-summary.json`
- `wandb/run-20260319_054056-qju56ygl/files/wandb-summary.json`
- `wandb/run-20260321_104506-nsniycxi/files/wandb-summary.json`
- `wandb/run-20260318_104331-5y90hbdc/files/config.yaml`
- `wandb/run-20260319_054056-qju56ygl/files/config.yaml`
- `wandb/run-20260321_104506-nsniycxi/files/config.yaml`

Representative rendering panels:

- `wandb/run-20260318_104331-5y90hbdc/files/media/images/unseen_object_with_unseen_envir_18000_440d03fd9ee776cb47b2.png`
- `wandb/run-20260319_054056-qju56ygl/files/media/images/unseen_object_with_unseen_envir_18000_440d03fd9ee776cb47b2.png`
- `wandb/run-20260321_104506-nsniycxi/files/media/images/unseen_object_with_unseen_envir/result_18000_6459a9a920c3eee93ee2.png`
- `wandb/run-20260318_104331-5y90hbdc/files/media/images/unseen_object_with_random_area_light_condition_17999_440d03fd9ee776cb47b2.png`
- `wandb/run-20260319_054056-qju56ygl/files/media/images/unseen_object_with_random_area_light_condition_17999_440d03fd9ee776cb47b2.png`
- `wandb/run-20260321_104506-nsniycxi/files/media/images/unseen_object_with_random_area_light_condition/result_17999_6cb3ff02b186604b42e7.png`
低风险、最值得先试

把前景阈值 foreground_background_threshold 从现在的 0.98 再放宽一点到 0.95-0.97。你现在高光分数图先被前景 mask 裁了一次，再缩放到 latent 尺度，neural_gaffer_training.py (line 819) 和 neural_gaffer_training.py (line 847)。0.98 对面光源这种边缘宽、半影重的区域有点太硬，容易把有效高光边界切掉。
把 num_validation_batches 从 2 提到 4-8，不然 random_area_light_condition 这条线很容易被小样本波动放大。当前配置在 neural_gaffer_training_gpu1_highlight.txt (line 8)。
给 image_space_constraint_weight 和 highlight_loss_weight 做 warmup，而不是一上来就全强度。现在它们固定是 0.1 和 2.0，见 neural_gaffer_training_gpu1_highlight.txt (line 28) 和 neural_gaffer_training_gpu1_highlight.txt (line 29)。更稳的做法是前 2k-4k step 从小到大爬升，让模型先学基础 relighting，再学高光细节。
更可能补随机面光源的改动

不要只按全局亮度做高光定义，改成“亮度 + 局部对比”或“亮度 + 高频残差”的混合分数。你现在的核心还是 luminance > threshold_map，代码在 neural_gaffer_training.py (line 812) 到 neural_gaffer_training.py (line 837)。这对宽而软的 area-light specular lobe 还是偏吃亏。
在算 quantile 前先对亮度图做一个轻微 blur，再取分位数。这样高光区域会从“碎亮点”变成“连贯亮斑”，通常更适合面光源。
对 lighting_idx_cond == -1 的样本单独用更宽松的 mask 或更高的 loss 权重，而不是所有样本一刀切。数据里这个分支很明确，在 dataset_relighting_training.py (line 290) 和 dataset_relighting_training.py (line 293)。
从鲁棒性角度再补两刀

给 condition image 加轻量光照扰动，比如 brightness/gamma jitter，目的是减少模型对输入打光外观的死记硬背。你现在随机面光源 split 的问题，本质上还是 condition image OOD。
把 random_lighting_condition_prob=0.4 改成 curriculum，而不是全程固定。比如前 5k 用 0.2，中段升到 0.4，后段到 0.5。这样通常比一开始就高比例更稳。
改 best checkpoint 选择口径，不要只看 held-out mean PSNR，可以加一个 floor，要求 random_area_light_condition 不能低于某个阈值，否则不记为 best。这样不会出现“主指标更好，但蓝线掉太多”的 best model。
如果你要我给一个最实用的实验顺序，我会这样排：

q=0.88，foreground_background_threshold=0.96，num_validation_batches=6。
在第 1 组上，加 image_space_constraint_weight 和 highlight_loss_weight 的 3k step warmup。
还是在第 1 组上，把 lighting_idx_cond == -1 的样本单独放宽 highlight mask。
如果你愿意，我可以直接把第 1 组先改进配置里，或者继续把 warmup 逻辑也一起写进训练脚本。


实际改动

highlight_quantile: 0.90 -> 0.88
原因是当前高光区域还是偏稀，先小幅放宽，比直接降到 0.85 更稳。
foreground_background_threshold: 0.98 -> 0.96
这个现在主要走默认值，建议显式写进配置。它能减少面光源下前景边缘和宽高光被截掉的问题。
num_validation_batches: 2 -> 6
这不增加训练显存，只会增加一点验证时间，但能显著降低蓝线波动，避免我们误判。
max_train_steps 继续保持 18k-20k
这轮不需要再拉长。



这一轮先保持不变

给 image_space_constraint_weight 和 highlight_loss_weight 加一个前 3k step 的线性 warmup。

random_lighting_condition_prob = 0.4
先别升到 0.5，否则训练分布变化太大，不利于判断这次改动到底有没有用。
image_space_constraint_weight = 0.1
highlight_loss_weight = 2.0
highlight_gamma = 2.0

这一轮先不要动

更激进的 q=0.85
random_lighting_condition_prob=0.5
按 random_lighting 样本单独加权
改成更复杂的局部对比/模糊高光定义

时间步退火 (Timestep Annealing) —— 让物理约束慢慢生效：
不要在所有的 $t$ 步都加上全额的 image_space_loss。
在 $t$ 很大的时候（早期，图像很模糊），物理约束的权重应该接近 0。只有当 $t$ 比较小（比如 $t < 300$，预测出的 $x_0$ 已经很清晰、边缘很锐利了）时，再逐渐增大高光 Loss 的权重。这样能避免早期模糊的面积光被阈值切碎。

cd /4T/CXY/Neural_Gaffer
LOG=logs/neural_gaffer_training_gpu1_highlight/restart_gpu0_$(date +%Y%m%d_%H%M%S).log
nohup bash -lc '
source /4T/conda_envs/neural_gaffer_5090/bin/activate
cd /4T/CXY/Neural_Gaffer
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun --standalone --nnodes=1 --nproc_per_node=1 neural_gaffer_training.py \
  --config configs/neural_gaffer_training_gpu1_highlight.txt
' > "$LOG" 2>&1 < /dev/null & disown
echo "$LOG"


cd /4T/CXY/Neural_Gaffer
tail -f "$(ls -1t logs/neural_gaffer_training_gpu1_highlight/restart_gpu0_*.log | head -n 1)"
