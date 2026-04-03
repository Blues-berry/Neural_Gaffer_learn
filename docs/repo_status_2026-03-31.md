# 仓库现状与清理说明（2026-03-31）

这份文档用于把“当前仓库真实状态”和“论文/日志文档里的引用口径”分开说明，避免再把历史 run、当前本地缓存、导出资产和计划文档混在一起。

## 1. 哪个目录才是当前主仓库

当前活跃工作树是：

- `/4T/CXY/Neural_Gaffer`

同级目录中容易混淆、但不属于当前主仓库的还有：

- `/4T/CXY/Neural_Gaffer_original`
  - 更接近上游原始版本和早期数据组织。
- `/4T/CXY/Neural_Gaffer_original_main_baseline`
  - 主 baseline 对照副本。
- `/4T/CXY/wandb`
  - 工作区顶层残留的独立 W&B 目录，不是本仓库内的规范入口。
- `/4T/CXY/anaconda3_backup`
  - 环境备份，与项目代码无关。

## 2. 当前代码主线

本仓库现在的主要工作可以分成四条线：

### 2.1 训练与推理主线

- `neural_gaffer_training.py`
- `neural_gaffer_inference_real_data.py`
- `neural_gaffer_inference_objaverse_3d.py`
- `parse_args.py`

`parse_args.py` 已经支持三种配置入口：

- `--config`
- `--method_config`
- `--data_config`

其中 `configs/neural_gaffer_training_gpu1_highlight.txt` 文件头已经明确写了当前更推荐的工作流：

- `configs/methods/highlight_hybrid_probe.txt`
- `configs/datasets/current_local_official.txt`

### 2.2 数据与子数据集整理主线

- `scripts/precheck_subdatasets.py`
- `scripts/validate_ready_datasets.py`
- `scripts/build_dataset_union.py`
- `scripts/build_validation_union.py`
- `scripts/assess_dataset_quality.py`
- `scripts/make_dataset_showcase.py`

对应的可训练子数据集打包说明已经存在于：

- `subdataset_backups/ready_20260328/README.md`

### 2.3 对比图与论文资产导出主线

- `scripts/create_relighting_comparison_manifest.py`
- `scripts/run_ours_on_comparison_manifest.py`
- `scripts/run_dilightnet_on_comparison_manifest.py`
- `scripts/export_relighting_comparison_assets.py`
- `scripts/build_relighting_comparison_panel.py`
- `scripts/run_checkpoint_panel_suite.py`

### 2.4 论文写作主线

- `docs/cadgraphics_template_cn.tex`
- `docs/paper_plan_cn.md`
- `docs/3.27三十八届图形仿真大会_experiment_and_figure_plan_cn.md`
- `docs/figures/`

## 3. 当前本地实际存在的关键产物

### 3.1 本地 checkpoint

当前最明确、最成体系的一组 checkpoint 位于：

- `logs/neural_gaffer_training_gpu1_highlight/`

本地可见的关键检查点包括：

- `checkpoint-10000`
- `checkpoint-74000`
- `checkpoint-75000`
- `checkpoint-76000`
- `checkpoint-77000`
- `checkpoint-78000`
- `checkpoint-79000`
- `checkpoint-80000`

这组 checkpoint 也已经被用于本地 sweep：

- `configs/comparison_suites/gpu1_highlight_checkpoint_sweep.json`
- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/`

### 3.2 本地 sweep 面板与导出资产

下面这些内容在当前仓库里是实际存在、可以直接核对的：

- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/ra_checkpoint_sweep_panel.json`
- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/uu_checkpoint_sweep_panel.json`
- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/ra_assets/`
- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/uu_assets/`
- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/us_assets/`
- `logs/relighting_comparison/gpu1_highlight_checkpoint_sweep/*_preds/`

其中面板 JSON 明确记录了 sweep 使用的是：

- suite: `configs/comparison_suites/gpu1_highlight_checkpoint_sweep.json`
- succeeded: `10k, 74k, 75k, 76k, 77k, 78k, 79k, 80k`
- device: `cuda:1`

### 3.3 本地 W&B 缓存

当前仓库内 `wandb/` 目录中，至少可以直接看到并与论文主线相关的本地缓存 run：

- `qju56ygl`
- `nsniycxi`
- `jbhdfvfc`
- `spuru5qy`
- `fi04f78e`
- `iepfybfu`
- `xkmlb19f`

另外还有一些更近期 run，例如：

- `973gqah7`
- `c66bmmrj`
- `dfsyx2cg`
- `4a9yd94d`
- `plyikmtp`
- `seyftjac`
- `2t3z4gz4`
- `keq5imbi`
- `ptsukrol`
- `o0xbj9sa`

这些 run 在文档中不一定都已经被解释清楚，因此不应默认视为“论文主结论 run”。

### 3.4 历史 baseline 的真实状态

论文和规划文档中经常引用：

- baseline run `7cn19b1e`

当前核对结果是：

- 该 run 被作为历史正确 baseline 口径继续使用。
- 它没有出现在当前仓库的本地 `wandb/` 缓存目录中。
- 因此，凡是引用 `7cn19b1e` 的文档，都应理解为“历史引用值”而不是“当前仓库内可直接打开的本地 run”。

这也是此前“文档和实际不一致”的一个主要来源。

## 4. 当前目录应该怎么理解

### 4.1 `docs/`

这里现在既有论文主稿，也有阶段性研究记录、参考论文和图示。已经补了：

- `docs/README.md`
- `docs/paper.md`

以后建议把 `docs/` 内文件分为三类来读：

- 正式入口文档
- 阶段性实验记录
- 参考材料与归档模板

### 4.2 `logs/`

`logs/` 里混合了三种完全不同的东西：

- 真正的训练输出和 checkpoint
- 用于论文/汇报的导出资产
- 调试、中间结果和 smoke test 产物

因此不应再把 `logs/` 理解成一个单一语义的“训练日志目录”。

### 4.3 `training_data/` 与 `validation_data/`

这两个目录是本地数据入口，不是轻量仓库资源。它们与脚本、配置和导出资产耦合很深，但不适合作为 README 的主要说明对象。

## 5. 这次做了哪些清理

本次整理只做“低风险、可回溯、不会破坏实验结果”的清理：

1. 重写根 README，使其反映当前本地工作区而不是上游开源页。
2. 增加 `docs/README.md` 和本状态文档，统一入口。
3. 把旧的占位式 `docs/paper.md` 改成论文写作入口。
4. 将命名不合适的 `docs/stupid plan.md` 归档为 `docs/archive/legacy_plan_template.md`。
5. 删除仓库根目录和 `docs/` 中已跟踪的 LaTeX 中间构建垃圾文件，只保留源码与 PDF 预览。
6. 更新 `.gitignore`，避免这些 LaTeX 中间文件再次混入版本库。

## 6. 后续建议

如果继续整理，优先级建议如下：

1. 给论文主线 run 做一份单独的 `run -> 配置 -> 指标 -> 本地证据路径` 对照表。
2. 把 `logs/relighting_comparison/` 下明显重复的 `v1/v2/v3/...` 导出资产做一次人工保留策略。
3. 逐步把论文正文、图示清单、实验计划中的“历史引用值”和“本地可复现实物”彻底拆开。
