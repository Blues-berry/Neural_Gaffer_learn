# 论文内容规划（面向图形仿真大会）

目标：通过更清晰的论文叙事、方法论展开与贡献组织，提升论文深度与说服力；以现有证据对齐版实验为主线，明确补实验路径。

## 1. 叙事主线（Narrative Arc）
1. 问题定位：扩散式物体重光照在镜面材质与面光源条件下，高光监督失稳是核心 failure mode。
2. 关键原因：前景分离不稳 + 绝对亮度阈值不稳 + 条件图像异常光照干扰。
3. 方法核心：前景掩码感知的相对高光引导（统一框架而非多个小技巧拼接）。
4. 证据链：阶段性 ablation 表 + 代表性视觉结果 + 随机面光源（ra）重点定性。
5. 结论与边界：在不增加推理复杂度前提下稳定监督；仍需更强掩码/极端光照处理。

## 2. 方法论深化（写作重点）
将方法组织为一个“高光监督稳定化框架”，避免碎片化描述。

### 2.1 统一定义（建议结构）
- 前景优先：显式 mask/alpha 优先，白背景阈值仅作 fallback。
- 局部相对高光：在有效前景内构建相对亮度度量（difference/ratio），替代绝对亮度。
- 分位数阈值：前景内 quantile 估计高光阈值，稳住阈值漂移。
- 仅用于阈值估计的轻量模糊：稳定阈值，但最终高光得分仍在原图域计算。
- 图像空间高光引导：RGB 级辅助监督，不改变推理阶段结构。

### 2.2 强调“单一机制逻辑”
一句话定义法：
“在前景约束下，把高光定义从绝对亮度迁移到局部相对亮度，并用分位数+轻模糊稳住阈值，最终用 image‑space 分支提供高信息量监督。”

## 3. 贡献（建议 3 条，正文可展开）
1. 明确提出并分析扩散重光照在 specular/area-light 下的高光监督失稳问题，并给出可解释原因链。
2. 提出前景掩码感知的相对高光引导框架（前景优先 + 相对高光 + 分位数阈值 + blur-only + image-space），在不增加推理复杂度下提升稳定性。
3. 设计分阶段数据协议（official/official-2000/ecommerce/3D-FUTURE）区分方法收益、数据收益与跨域收益，便于评测与复现。

## 4. 实验与证据组织（以现有文档为主）

### 4.1 定量主表（证据对齐版）
使用现有 evidence‑aligned ablation 表（7cn19b1e / qju56ygl / nsniycxi / jbhdfvfc / spuru5qy / fi04f78e）。
写作策略：
- 用 `jbhdfvfc` 作为“主 held‑out 最强”锚点。
- 用 `spuru5qy` 强调 ra 鲁棒性提升（关键 failure mode 修复）。
- 用 `fi04f78e` 支撑“相对高光定义”合理性。

### 4.2 Clean ablation（补实验方向）
已完成 `abl00_base` / `abl01_imgspace_fixed` 作为“image‑space 分支必要性”证据。
计划：放入补实验表或附录，明确与主证据表区分。

### 4.3 其他 TBD 表格
需要补齐数值的部分（在模板里保持清晰标记）：
- 训练数据策略表（official-1000 / official-2000 / ecommerce 等）
- 跨域泛化表（official / ecommerce / 3D‑FUTURE）
- 效率变体表（Peak VRAM / steps per hour）

## 5. 图示规划（沿用现有槽位）
1. 问题定义图：baseline 在 ra/soft highlight 失败。
2. 方法流程图：foreground -> relative -> quantile + blur -> image‑space。
3. 主对比图：official 的 uu/us/ra，突出稳定性。
4. 跨域图：official/ecommerce/3D‑FUTURE 三列。
5. 效率对比图：image‑space / hybrid / pure probe。
6. 诊断图：highlight mask/weight 与结果图对照。

## 6. 关键论断与证据映射（写作时对齐）
| 论断 | 证据 |
| --- | --- |
| 固定阈值不稳 | qju56ygl 与 baseline 的 ra 降低 |
| quantile + blur 稳定 ra | spuru5qy 在 ra 提升 |
| 相对高光定义提升可解释性 | fi04f78e 保持 uu/us + 方法分析 |
| image-space 分支必要 | clean ablation：abl01 vs abl00 |

## 7. 需要补的材料清单（执行项）
- 补齐跨域与数据策略表格数值。
- 生成/选取跨域对比图与效率图。
- 补 VRAM 与 steps/hour 统计。
- 整理 clean ablation 表为附录或补实验。
- 更新作者信息、单位、联系方式。

## 8. 写作注意事项（提高深度）
- 避免“堆技巧”，强调“统一监督框架”。
- 失败模式要具体：哪些场景、怎么失败、为什么失败。
- ra 作为主困难划分，强调“高光宽波瓣 + 随机面光源”的意义。
- 把 hybrid/pure probe 明确成效率变体，而非主方法贡献。

## 9. 输出建议（目标顺序）
1. 完善 LaTeX 主稿的实验表与文字说明。
2. 补齐跨域与效率表格数值。
3. 填充/替换图示占位。
4. 最后统一润色摘要与结论（收敛主线）。
