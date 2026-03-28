# 可训练子数据集打包与划分说明（2026-03-28）

本目录是“可进行训练的子数据集”备份元数据。实际可训练数据已按四个类别打包放在：

- `/4T/CXY/Neural_Gaffer/logs/ready_subdatasets_20260328/`

该打包目录使用**软链接视图**组织数据（不复制大文件），并提供训练/验证所需的对象列表与验证填充结构。

## 1. 目录结构（每个子数据集）

每个子数据集目录包含：

- `images/`：训练图像的对象目录（软链接视图）
- `lighting/`：训练光照的 `LDR/`、`HDR_rescaled/`、`HDR_raw/`（软链接视图）
- `training_object_list.json`：训练对象列表（已排除 val_unseen）
- `val_seen_object_list.json` / `val_unseen_object_list.json`：验证对象列表
- `val_split.json`：划分统计
- `meta.json`：源路径与统计信息
- `val/`：验证集填充结构（软链接视图）
  - `val/images/{seen_lighting,unseen_lighting}/`
  - `val/lighting/{seen_lighting,unseen_lighting}/{LDR,HDR_rescaled,HDR_raw}/`

> 说明：`val/images/*` 目录下同时放置了 `val_seen_object_list.json` 与 `val_unseen_object_list.json`，以兼容 `dataset_relighting_training.py` 对 `dataset_type` 的读取方式。

## 2. 划分规则（初步版）

采用固定规则生成初始划分：

- 先按对象 ID 排序；
- `val_unseen` 取前 5%（至少 10 个）；
- `val_seen` 取紧随其后的 5%（至少 10 个）；
- 训练集 = 去掉 `val_unseen` 后的剩余对象（`val_seen` 仍保留在训练集）。

该规则已写入 `val_split.json`，便于后续替换为更合理的随机划分或人工挑选。

## 3. 训练/验证路径对齐（对应代码逻辑）

当前训练逻辑在 `neural_gaffer_training.py` / `dataset_relighting_training.py` 中依赖以下路径：

- `train_img_dir` + `train_lighting_dir`：训练数据
- `val_img_dir` + `val_lighting_dir`：验证数据
- `val_img_dir` 下必须有 `seen_lighting` / `unseen_lighting` 子目录

因此使用打包数据时推荐配置：

```
train_img_dir     = <pkg>/images
train_lighting_dir= <pkg>/lighting
val_img_dir       = <pkg>/val/images
val_lighting_dir  = <pkg>/val/lighting
```

## 4. 重要注意事项

- 本次打包仅包含**真正具备 LDR/HDR_rescaled 光照目录**的对象，因此数量可能小于状态文件中的 `complete_train_object_count`。
- `three_future` 中有部分对象缺少完整光照目录，已被排除。
- 所有打包目录中的 `.tar.gz` 为软链接打包，**不会复制大文件**。

## 5. 可用统计（以 READY_SUMMARY.json 为准）

请查看同目录下：

- `READY_SUMMARY.json`

该文件记录了：

- `trainable_count`：可训练对象总数（过滤后）
- `train_count_after_split`：排除 val_unseen 后的训练数量
- `val_seen_count` / `val_unseen_count`

