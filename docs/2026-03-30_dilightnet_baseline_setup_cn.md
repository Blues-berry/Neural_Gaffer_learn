# DiLightNet 基线独立环境与运行说明

这套流程的目标是：

- 不修改当前可跑的 `neural_gaffer_5090` 环境
- 尽量遵循 `DiLightNet` 官方仓库的依赖设置
- 直接对接当前项目已有的 comparison manifest 与面板生产线

## 独立环境

推荐通过单独的 conda 前缀环境完成安装：

```bash
bash scripts/setup_dilightnet_baseline_env.sh /4T/conda_envs/dilightnet_baseline
```

这一步会：

- 先从 `neural_gaffer_5090` clone 一个隔离副本
- 在副本中执行 `external/DiLightNet_full/requirements.txt`
- 做一次最小 import smoke test

如果你想完全按官方 README 的 Python/Torch/CUDA 组合从头建环境，而不是 clone 现有 env，可以这样：

```bash
SETUP_MODE=official bash scripts/setup_dilightnet_baseline_env.sh /4T/conda_envs/dilightnet_baseline
```

注意：当前机器上的脚本目录可能是 `noexec` 挂载，因此建议显式使用 `bash scripts/...` 调用，而不是直接 `./scripts/...`。

## 单个 manifest 运行

```bash
bash scripts/run_dilightnet_baseline_safe.sh \
  logs/relighting_comparison/ra_manifest.json \
  logs/relighting_comparison/dilightnet_preds_ra \
  --staging-dir logs/relighting_comparison/dilightnet_staging_ra \
  --steps 20 \
  --cfg 3.0 \
  --prompt ""
```

该脚本会自动：

- 使用独立环境中的 python
- 调用 `scripts/run_dilightnet_on_comparison_manifest.py`
- 使用官方 `external/DiLightNet_full/infer_img.py`
- 按当前项目约定输出到 `<root>/<object_id>/pred_image/<target_file>`

## 三个 split 连跑

```bash
bash scripts/run_dilightnet_comparison_all.sh \
  logs/relighting_comparison/dilightnet_preds \
  --steps 20 \
  --cfg 3.0 \
  --prompt ""
```

默认会顺序运行：

- `ra_manifest.json`
- `uu_manifest.json`
- `us_manifest.json`

## 对接面板导出

当 `DiLightNet` 结果按约定目录写好后，可继续沿用现有脚本：

```bash
python scripts/export_relighting_comparison_assets.py \
  --manifest logs/relighting_comparison/ra_manifest.json \
  --output-dir logs/relighting_comparison/ra_assets_with_dilightnet \
  --method-root dilightnet=logs/relighting_comparison/dilightnet_preds/ra \
  --method-root ours=logs/relighting_comparison/ours_preds_ra_v1

python scripts/build_relighting_comparison_panel.py \
  --assets-manifest logs/relighting_comparison/ra_assets_with_dilightnet/exported_assets_manifest.json \
  --output logs/relighting_comparison/ra_panel_with_dilightnet.png
```

## 当前策略

为了不搞崩现有训练环境，后续所有 `DiLightNet` 相关安装与运行都应限制在：

- `/4T/conda_envs/dilightnet_baseline`

而不是直接安装到：

- `neural_gaffer_5090`
- `neural-gaffer`
- `base`
