# Dataset Reference

本文件记录数据根、union 构建、same-batch 数据和清理边界。

## 资源根

- 当前项目数据: `training_data/`、`validation_data/`
- 迁移后的原始数据与 raw render: `external_data/neural_gaffer_original/`
- 3D/raw source: `external_data/neural_gaffer_original/external_sources/`
- Objaverse raw render jobs: `external_data/neural_gaffer_original/objaverse_jobs/`
- lighting domains: `external_data/neural_gaffer_original/objaverse_lighting_domains/`
- official baseline code: `external/official_neural_gaffer_baseline/`

## Dataset Union

```bash
cd /4T/CXY/Neural_Gaffer
/home/ubuntu/anaconda3/bin/python scripts/build_dataset_union.py --preset main
/home/ubuntu/anaconda3/bin/python scripts/build_dataset_union.py --preset full
/home/ubuntu/anaconda3/bin/python scripts/build_dataset_union.py --preset all_available
```

当前 `all_available` union:

- 输出: `logs/dataset_unions/full_current_original_official2000_ecommerce1000_3dfuture_landscape`
- 对象数: 3029
- 数据配置: `configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture_landscape_allready_plus_officialval.txt`

## Validation Union

常用验证根:

- `logs/dataset_validation_unions/all_ready_plus_official_20260403`

same-batch manifest 默认使用:

- raw roots: `external_data/neural_gaffer_original/objaverse_jobs/*/raw`
- lighting roots: `external_data/neural_gaffer_original/training_data/lighting/*`

## 覆盖环境变量

- `NEURAL_GAFFER_ORIGINAL_ASSETS_ROOT`
- `NEURAL_GAFFER_ORIGINAL_RENDER_SCRIPTS`
- `NEURAL_GAFFER_MODEL_CACHE_ROOT`
- `NEURAL_GAFFER_OFFICIAL_BASELINE_REPO`
- `NEURAL_GAFFER_OFFICIAL_CHECKPOINT_ROOT`

## 删除边界

可以清理:

- `logs/smoke/`
- `logs/dataset_unions_legacy_pre_migration_*/`
- 纯 `.log`、`.stdout`、`.stderr`、`.tmp`
- 不再被 suite 使用的重复 `logs/neural_gaffer_training_*` exported pipeline
- `__pycache__/` 和 `*.pyc`

不要清理:

- `model_weights/neural_gaffer_model_cache/`
- `external_data/neural_gaffer_original/`
- `logs/dataset_unions/`
- `logs/dataset_validation_unions/`
- `logs/ready_subdatasets_20260328/`
- `effects/` 下的论文图、表和评测结果

删除 `/4T/CXY/Neural_Gaffer_original` 前必须确认:

```bash
pgrep -af /4T/CXY/Neural_Gaffer_original
```

没有任何 render、Blender、preprocess、monitor 进程仍在使用旧兼容路径。
