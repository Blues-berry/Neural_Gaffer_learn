# 2026-03-31 CAD/Graphics 投稿口径、实验矩阵与时间线

这份文档不是研究随笔，而是面向 `CAD/Graphics 2026` 的执行版投稿方案。

目标只有一个：

- 把“当前已经成型的方法线”
- “当前本地已存在的数据配置”
- “接下来到投稿截止前必须完成的实验与写作动作”

收拢成一套可以直接照着推进的口径。

状态说明（2026-03-31）：

- 本文档中的截止时间以官方投稿页当前显示为准：
  - abstract: `2026-05-14`
  - full paper: `2026-05-28`
  - page limit: `12 pages`
  - review policy: `double-blind`
- 当前仓库内实际存在的数据配置和 union 目录，已经核对过：
  - `configs/datasets/main_current_original_official_ecommerce.txt`
  - `configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture.txt`
  - `logs/dataset_unions/main_current_original_official_ecommerce`
  - `logs/dataset_unions/full_current_original_official2000_ecommerce1000_3dfuture`

---

## 1. 论文应该怎么定义

### 1.1 最推荐的问题定义

不要把论文写成：

- `Neural Gaffer 的若干训练技巧优化`

更合适的写法是：

**Diffusion-based object relighting 在 specular / area-light 条件下，高光监督容易因为前景裁剪不稳、绝对亮度阈值不稳和宽波瓣高光而失真；我们提出一种 foreground-mask-aware relative highlight guidance，在不增加推理复杂度的前提下稳定高光监督，并提升 object relighting 的鲁棒性与跨域泛化。**

### 1.2 一句话版本

论文最核心的一句话建议固定成：

**We stabilize highlight supervision for diffusion-based object relighting by making highlight definition foreground-aware, relative, and quantile-driven, without changing the inference-time relighting pipeline.**

### 1.3 边界要说清楚

当前最适合讲的任务边界是：

1. `single-image`
2. `object-centric`
3. `environment-map-conditioned relighting`
4. 重点关注 `specular / area-light robustness`

不要把口径写成任意复杂场景 relighting。

---

## 2. 论文标题建议

### 2.1 主标题候选

最推荐：

**Foreground-Mask-Aware Relative Highlight Guidance for Robust Object Relighting**

优点：

1. 和当前代码主线高度对齐
2. 不依赖 probe 这条还不够稳的辅线
3. 直接把 novelty 落在“高光监督原则”上

### 2.2 更保守的标题

如果担心 reviewer 对 “guidance” 一词理解不一，可以用：

**Stabilizing Highlight Supervision for Diffusion-based Object Relighting**

副标题可选：

**with Foreground-Mask-Aware Relative Highlight Guidance**

### 2.3 不建议当主标题的词

当前阶段不建议把这些词放进主标题：

1. `hybrid probe distillation`
2. `latent highlight probe`
3. `efficient`
4. `physics-inspired`

原因：

1. probe 线现在更适合做效率变体
2. “physics-inspired” 会把 reviewer 预期抬得太高
3. 主方法真正稳定成立的还是 foreground-aware highlight supervision

---

## 3. 摘要口径

### 3.1 英文摘要草案

下面这版是当前最适合往 LaTeX 主稿里推进的摘要基线：

```text
Diffusion-based object relighting has recently shown promising visual quality under environment-map guidance, yet its supervision becomes unstable in specular and area-light settings. In these cases, highlight regions are spatially sparse, sensitive to foreground extraction, and poorly characterized by fixed absolute-brightness thresholds, which often leads to over-exposed or over-smoothed relighting results. We present a foreground-mask-aware relative highlight guidance framework for robust object relighting. Our method prioritizes explicit foreground masks and uses white-background thresholding only as a fallback, defines highlights in a local relative-intensity space, estimates thresholds with foreground-only quantiles, and applies blur only for threshold estimation while preserving the final score in the original image domain. We further retain an image-space highlight guidance branch to provide direct RGB-level supervision without changing the inference-time relighting pipeline. To evaluate both in-domain quality and cross-domain generalization, we organize a multi-domain object-relighting protocol spanning official data, an ecommerce-style product domain, and a 3D-FUTURE furniture domain. Experiments show that the proposed framework improves held-out relighting quality and yields more stable highlight responses under difficult specular and random area-light conditions.
```

### 3.2 中文摘要压缩版

如果先写中文提纲，建议压成下面这版：

- 先指出 failure mode：specular / area-light 下高光监督失稳
- 再指出原因链：前景提取、绝对阈值、宽高光波瓣
- 然后写方法：foreground-aware、relative、quantile、blur-only、image-space
- 最后写评测协议：official / ecommerce / 3D-FUTURE

### 3.3 摘要里不要写什么

不建议在摘要里主打：

1. `hybrid probe distillation`
2. `8bit Adam`
3. `guard / early stop / best checkpoint`
4. 各种实现侧 warmup 技巧

这些内容可以在实验或附录里出现，但不该支配摘要。

---

## 4. 主贡献与辅贡献

### 4.1 主贡献建议固定为 3 条

#### 贡献 1

我们识别并分析了 diffusion-based object relighting 在 specular / area-light 条件下的一个关键失败模式：

1. 前景提取不稳会污染高光监督
2. 固定绝对亮度阈值难以覆盖宽而软的高光波瓣
3. 随机面光源与异常 condition lighting 会进一步放大这一问题

#### 贡献 2

我们提出 `foreground-mask-aware relative highlight guidance`：

1. 显式 alpha / foreground mask 优先
2. 在局部相对亮度域定义高光
3. 只在前景内做 quantile threshold
4. 模糊只用于 threshold estimation
5. image-space highlight branch 提供高信息量 RGB 监督

#### 贡献 3

我们建立面向目标域与跨域泛化的 object-relighting evaluation protocol：

1. `official` 作为标准协议
2. `ecommerce` 作为产品域增强与目标域评测
3. `3D-FUTURE` 作为跨域测试域

### 4.2 辅贡献

下面这些可以写，但不要写成主贡献：

1. `hybrid probe distillation`
2. `latent highlight probe`
3. 数据预检脚本
4. 训练稳定性工程细节

---

## 5. 最终建议的数据协议

### 5.1 数据角色先固定

当前最稳妥的角色划分如下：

#### `official`

角色：

1. baseline 所在标准协议
2. clean ablation 的唯一固定训练集
3. 主 held-out 测试口径

#### `ecommerce`

角色：

1. 目标域增强训练集
2. 产品类 object relighting 的目标测试域
3. 高光/商品材质表现的重点 qualitative 域

#### `3D-FUTURE`

角色：

1. 第一阶段：`test-only cross-domain benchmark`
2. 第二阶段：如果时间充足，再作为 supplementary train domain

#### `landscape`

角色：

1. robustness / failure analysis
2. 当前不进入主训练

### 5.2 当前主训练集

结合当前仓库里已经存在的配置与 union，主训练集建议先固定为：

- config: [main_current_original_official_ecommerce.txt](/4T/CXY/Neural_Gaffer/configs/datasets/main_current_original_official_ecommerce.txt)
- union: [main_current_original_official_ecommerce](/4T/CXY/Neural_Gaffer/logs/dataset_unions/main_current_original_official_ecommerce)

它的意义不是“最终全量最强”，而是：

1. 最容易把论文主线做扎实
2. 同时保留原始分布和商品高光目标域
3. 便于把 3D-FUTURE 留作跨域测试，不把解释搅浑

### 5.3 备用全量训练集

如果主线稳定、时间也够，再训练一条更完整的增强模型：

- config: [full_current_original_official2000_ecommerce1000_3dfuture.txt](/4T/CXY/Neural_Gaffer/configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture.txt)
- union: [full_current_original_official2000_ecommerce1000_3dfuture](/4T/CXY/Neural_Gaffer/logs/dataset_unions/full_current_original_official2000_ecommerce1000_3dfuture)

这条更适合在论文里扮演：

1. supplementary full-data model
2. scaling experiment
3. “更多域训练是否继续提升”的补充结果

不建议一开始就把它当主模型。

---

## 6. 最终建议的实验矩阵

### 6.1 表 1：方法 clean ablation

训练固定：

1. `official-1000`

建议行：

1. `Baseline`
2. `+ Image-space highlight guidance`
3. `+ Foreground-mask-aware supervision`
4. `+ Foreground quantile threshold`
5. `+ Blur-only threshold estimation`
6. `+ Local-relative highlight`
7. `+ Full main method`

指标：

1. `PSNR`
2. `SSIM`
3. `LPIPS`
4. `uu / us / ra / tu / train`

作用：

1. 明确证明主方法不是若干 trick 堆起来的偶然增益

### 6.2 表 2：高光专项指标表

这张表建议单独存在，不和全局指标混在一起。

推荐子集：

1. `official hard subset`
2. `ra-hard`
3. `ecommerce specular subset`

指标建议：

1. `highlight_psnr`
2. `highlight_rmse`
3. `highlight_mask_iou`
4. `highlight_area_abs_error`
5. `highlight_saturated_ratio_abs_error`
6. `highlight_p95_luma_abs_error`
7. `lpips_highlight_crop`

当前仓库已经有对应脚本基础：

- [evaluate_highlight_metrics_on_assets_manifest.py](/4T/CXY/Neural_Gaffer/scripts/evaluate_highlight_metrics_on_assets_manifest.py)

### 6.3 表 3：数据策略表

建议比较：

1. `official`
2. `main_current_original_official_ecommerce`
3. `full_current_original_official2000_ecommerce1000_3dfuture`

这张表回答：

1. 增加 `ecommerce` 是否能帮助目标域
2. 增加 `3D-FUTURE` 是否进一步提高泛化
3. 方法收益与数据收益分别有多大

### 6.4 表 4：跨域泛化表

训练先固定主模型。

测试建议包含：

1. official held-out
2. ecommerce held-out
3. 3D-FUTURE held-out

作用：

1. 证明方法不是只在训练域内涨
2. 让 `3D-FUTURE` 先作为 OOD 域站住脚

### 6.5 表 5：效率变体表（可选）

如果篇幅还够，再放：

1. image-space main method
2. hybrid probe distillation
3. pure latent replacement

指标：

1. `PSNR/LPIPS`
2. `Peak VRAM`
3. `steps/hour`

这张表是 bonus，不是主线必需项。

---

## 7. 图示规划

### 7.1 必须有的 4 组图

#### 图 1：问题定义图

内容：

1. baseline 在 `ra` 或宽高光 case 下的 failure
2. 典型现象：clipping、过散、位置漂移、材质发灰

#### 图 2：方法流程图

建议结构：

1. foreground mask priority
2. local relative highlight
3. foreground quantile threshold
4. blur-only threshold estimation
5. image-space highlight guidance

#### 图 3：主对比图

列建议：

1. Input
2. Baseline
3. Ours
4. GT
5. Target lighting

重点样本：

1. `uu`
2. `us`
3. `ra`

#### 图 4：高光局部放大 + 诊断图

建议并排放：

1. highlight crop
2. GT highlight mask
3. pred highlight mask
4. abs diff heatmap

### 7.2 可选图

如果篇幅允许，再加：

1. 跨域图：official / ecommerce / 3D-FUTURE
2. 效率图：main vs hybrid vs replacement
3. 稳定性曲线：同物体跨 lighting 的高光面积/峰值变化

---

## 8. 时间线

下面这个时间线是按官方页面当前显示的截止日期倒排的。

### 8.1 阶段 A：问题与协议冻结

时间：

- `2026-03-31` 到 `2026-04-06`

目标：

1. 冻结论文主方法名
2. 冻结主训练集与测试协议
3. 冻结 clean ablation 行定义

本阶段必须完成：

1. 选定主标题
2. 把摘要草案写进 LaTeX
3. 明确 `main_current_original_official_ecommerce` 是主训练线
4. 明确 `3D-FUTURE` 第一阶段只做测试域

### 8.2 阶段 B：核心实验出表

时间：

- `2026-04-07` 到 `2026-04-20`

目标：

1. 跑完 clean ablation 最小矩阵
2. 跑出主训练集上的核心结果
3. 生成初版主表和高光专项表

本阶段优先级：

1. `official-1000 clean ablation`
2. `main_current_original_official_ecommerce` 主模型
3. highlight-specific metrics

### 8.3 阶段 C：跨域与图示

时间：

- `2026-04-21` 到 `2026-05-05`

目标：

1. 跑完 official / ecommerce / 3D-FUTURE 测试
2. 选出论文主图
3. 做完高光局部放大图和诊断图

本阶段必须完成：

1. 跨域泛化表
2. 主对比 panel
3. highlight crop figure

### 8.4 阶段 D：摘要提交前收口

时间：

- `2026-05-06` 到 `2026-05-13`

目标：

1. 摘要和题目冻结
2. 作者名单冻结
3. 引言、方法、实验的大结构稳定

注意：

- 根据官方页面，`2026-05-14` 摘要提交后不能再改作者列表

### 8.5 阶段 E：全文冲刺

时间：

- `2026-05-15` 到 `2026-05-27`

目标：

1. 完成全文英文润色
2. 补最后缺口实验
3. 检查双盲合规
4. 完成 12 页压缩

最终截止：

- `2026-05-28 23:59 GMT`

---

## 9. 如果时间不够，优先砍什么

为了避免最后变成“大而乱”，建议按下面顺序删减。

### 第一优先级保留

一定要保住：

1. 主问题定义
2. clean ablation
3. 主模型结果
4. 高光专项指标表
5. 一组最强的 qualitative figure

### 第二优先级可降级

如果时间不够，可以降级成附录或删除：

1. `hybrid probe distillation`
2. full-data train with `+3D-FUTURE`
3. 效率表
4. 大规模超参扫参

### 第三优先级先不碰

当前最不建议继续花主线预算的是：

1. 新 backbone
2. 更多 probe 变体
3. landscape 主训练
4. 过度扩写 physics story

---

## 10. 从现在开始最值钱的 6 个动作

1. 把主标题和摘要固定到 LaTeX 主稿里
2. 把 clean ablation 的 6 到 7 行实验顺序冻结
3. 先跑 `main_current_original_official_ecommerce`
4. 用高光专项脚本补一张独立的 highlight metrics 表
5. 把 `3D-FUTURE` 先做成 test-only cross-domain result
6. 在 `2026-05-14` 前把题目、作者和摘要彻底冻结

---

## 11. 一句话决策

如果只能选一条最稳路线，那就是：

**主方法写 foreground-mask-aware relative highlight guidance，主训练先用 `main_current_original_official_ecommerce`，`3D-FUTURE` 先做跨域测试，不急着并入训练；先把 clean ablation、高光专项指标和跨域证据链做扎实，再考虑 full-data model 和 probe 效率线。**
