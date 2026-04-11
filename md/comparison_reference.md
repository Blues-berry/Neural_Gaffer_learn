# Comparison Reference

本文件记录 comparison suite、same-batch 出图和论文 panel 的当前入口。

## 推荐 Suite

- 单模型 smoke: `configs/comparison_suites/ours_single_local_ssd_v2.json`
- baseline vs ours: `configs/comparison_suites/minimal_baseline_vs_ours_full_local.json`
- official1000 baseline vs ours: `configs/comparison_suites/official1000_baseline_vs_ours_full_local.json`
- realchain substitute: `configs/comparison_suites/foreground_highlight_realchain_substitutes_0409.json`
- validation same-batch: `configs/comparison_suites/validation_samebatch_0407_baseline_vs_ours.json`

所有长期 checkpoint 路径应指向 `model_weights/neural_gaffer_model_cache/`，不要再指向 `logs/` 或 `/dev/shm/`。

## Same-Batch Manifest

```bash
cd /4T/CXY/Neural_Gaffer
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/python scripts/create_samebatch_validation_one_hdri_manifest.py \
  --validation-root logs/dataset_validation_unions/all_ready_plus_official_20260403 \
  --output-dir logs/samebatch/run_YYYYMMDD/manifests \
  --limit-objects 50 \
  --shard-size 50 \
  --fast-assume-native-size
```

生成:

- `same_batch_manifest_native512.json`
- `same_batch_audit.json`
- `manifest_summary.json`
- `shards/manifest_shard_*.json`

## Run Checkpoint Panel Suite

```bash
cd /4T/CXY/Neural_Gaffer
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/python scripts/run_checkpoint_panel_suite.py \
  --manifest logs/samebatch/run_YYYYMMDD/manifests/same_batch_manifest_native512.json \
  --suite configs/comparison_suites/ours_single_local_ssd_v2.json \
  --pred-root logs/samebatch/run_YYYYMMDD/predictions \
  --assets-dir logs/samebatch/run_YYYYMMDD/assets \
  --panel-output logs/samebatch/run_YYYYMMDD/panels/samebatch_panel.png \
  --device cuda:1 \
  --resolution 256 \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --tile-size 256 \
  --skip-existing
```

该脚本会依次执行:

- `scripts/run_ours_on_comparison_manifest.py`
- `scripts/export_relighting_comparison_assets.py`
- `scripts/build_relighting_comparison_panel.py`

## Paper Panel 资产

最终 panel 和资产 manifest 位于:

- `assets/exported_assets_manifest.json`
- `panels/*.png`
- `panels/*.json`

如果需要 highlight zoom 或 sorted panel，再基于 exported assets 调用:

- `scripts/build_highlight_zoom_panel.py`
- `scripts/build_sorted_suite_panels.py`
- `scripts/build_grouped_highlight_panels.py`
