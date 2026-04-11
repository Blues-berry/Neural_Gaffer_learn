# Neural_Gaffer Workspace

本仓库已经不是单纯的上游开源代码镜像，而是一个持续演化的本地研究工作区。当前内容同时包含：

- Neural Gaffer 主训练与推理代码
- 高光监督/频域辅助等方法试验
- 数据子集整理与质量检查脚本
- 对比图导出、面板拼接与 checkpoint sweep 工具
- 论文写作材料、图示与参考文献

原始项目论文与主页仍然适用：

- Project Page: <https://neural-gaffer.github.io/>
- Paper: <https://arxiv.org/abs/2406.07520>

## 当前仓库最重要的事实

- 当前活跃仓库就是本目录 `Neural_Gaffer/`。
- 同级目录 `../Neural_Gaffer_original/` 现在只作为原始代码工作树和正在运行的兼容入口保留；完整数据、raw render、外部 3D 源文件和原始 W&B/日志归档已经收敛到 `external_data/neural_gaffer_original/`。
- 官方 baseline worktree 已迁入 `external/official_neural_gaffer_baseline/`；旧 `../Neural_Gaffer_original_main_baseline/` 兼容软链接已经删除。
- 当前 README 只保留总入口；训练、推理、对比、消融和数据集的可执行参考请优先看 `docs/*_reference.md`。
- 论文文档里频繁引用的 baseline `7cn19b1e` 是历史 W&B 结果口径，并不在当前本地 `wandb/` 缓存中；本地可核对的 run、checkpoint 与导出资产已经在状态文档中单独说明。

## 当前常用入口

### 1. 训练

- 主训练脚本：`neural_gaffer_training.py`
- 参数定义：`parse_args.py`
- 兼容旧工作流的总配置：`configs/neural_gaffer_training_gpu1_highlight.txt`
- 当前更推荐的拆分式配置入口：
  - `configs/methods/highlight_hybrid_probe.txt`
  - `configs/datasets/current_local_official.txt`

示例：

```bash
accelerate launch --config_file configs/1_16fp.yaml neural_gaffer_training.py \
  --method_config configs/methods/highlight_hybrid_probe.txt \
  --data_config configs/datasets/current_local_official.txt
```

### 2. 推理与对比图

- 单图/真实图像推理：`neural_gaffer_inference_real_data.py`
- 3D/Objaverse 风格推理：`neural_gaffer_inference_objaverse_3d.py`
- 生成对比清单：`scripts/create_relighting_comparison_manifest.py`
- 导出统一资产：`scripts/export_relighting_comparison_assets.py`
- 拼接论文式面板：`scripts/build_relighting_comparison_panel.py`
- 批量 checkpoint sweep：`scripts/run_checkpoint_panel_suite.py`

### 3. 论文与图示

- 文档索引：`docs/README.md`
- 仓库现状与日志说明：`docs/repo_status_2026-03-31.md`
- 论文写作入口：`docs/paper.md`
- 中文主稿：`docs/cadgraphics_template_cn.tex`
- 图示与裁剪图：`docs/figures/`

## 目录说明

| 路径 | 作用 |
| --- | --- |
| `configs/` | 训练、数据协议、对比 suite 配置 |
| `dataset/` | 数据加载与前景掩码相关逻辑 |
| `scripts/` | 预处理、验证、导出、对比图、外部基线脚本 |
| `docs/` | 论文草稿、实验记录、图示、参考 PDF、归档模板 |
| `logs/` | 当前训练输出、导出资产、中间实验产物 |
| `model_weights/` | 论文复现和后续扩展需要保留的 checkpoint/cache |
| `training_data/` | 本地训练数据入口 |
| `validation_data/` | 本地验证数据入口 |
| `external/` | 外部基线或第三方代码副本 |

## 推荐阅读顺序

1. `docs/training_reference.md`
2. `docs/inference_reference.md`
3. `docs/comparison_reference.md`
4. `docs/ablation_reference.md`
5. `docs/dataset_reference.md`
6. `docs/paper.md`

## 备注

- `logs/`、`training_data/`、`validation_data/`、`wandb/` 大多属于本地工作产物，体量大、版本杂，仓库里默认不把它们当作稳定 API。
- 本次清理保留了实验结果本体，只对入口文档、说明口径和 LaTeX 构建垃圾做了整理。
