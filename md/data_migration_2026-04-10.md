# Data Migration 2026-04-10

This note records the local consolidation that moved Neural Gaffer data and baseline resources under the active workspace.

## New Resource Roots

- Original data assets: `/4T/CXY/Neural_Gaffer/external_data/neural_gaffer_original`
- Official baseline code: `/4T/CXY/Neural_Gaffer/external/official_neural_gaffer_baseline`
- Persistent model cache: `/4T/CXY/Neural_Gaffer/model_weights/neural_gaffer_model_cache`
- Current generated dataset unions: `/4T/CXY/Neural_Gaffer/logs/dataset_unions`
- Previous generated dataset unions were removed after validation because the current union metadata is preserved under `logs/dataset_unions/`.

## Moved From `Neural_Gaffer_original`

These former top-level directories were moved into `external_data/neural_gaffer_original/`, and compatibility symlinks were left at the old paths:

- `training_data`
- `validation_data`
- `external_sources`
- `objaverse_jobs`
- `objaverse_lighting_domains`
- `subdataset`
- `logs`
- `wandb`
- `test`

The tracked planning and rendering pipeline inputs were copied, not removed from the original worktree:

- `external_data_plan` copied to `external_data/neural_gaffer_original/plans/external_data_plan`
- `objaverse_subsets` copied to `external_data/neural_gaffer_original/manifests/objaverse_subsets`
- `scripts/Objavarse_rendering` copied to `external_data/neural_gaffer_original/rendering_pipeline/Objavarse_rendering`

## Updated Entrypoints

The following scripts now default to the consolidated paths and can still be overridden by environment variables:

- `scripts/build_dataset_union.py`
- `scripts/build_same_batch_manifest.py`
- `scripts/create_samebatch_validation_one_hdri_manifest.py`
- `scripts/precheck_subdatasets.py`
- `scripts/run_highlight_test_samples_pipeline_0407.py`
- `scripts/run_official_demo_on_comparison_manifest.py`
- `scripts/run_official_demo_gallery.py`
- `scripts/run_validation_samebatch_pipeline_0407.py`
- `scripts/materialize_model_cache_from_suite.py`
- `scripts/export_relighting_comparison_assets.py`

Supported overrides:

- `NEURAL_GAFFER_ORIGINAL_ASSETS_ROOT`
- `NEURAL_GAFFER_ORIGINAL_RENDER_SCRIPTS`
- `NEURAL_GAFFER_OFFICIAL_BASELINE_REPO`
- `NEURAL_GAFFER_OFFICIAL_CHECKPOINT_ROOT`
- `NEURAL_GAFFER_MODEL_CACHE_ROOT`

## Deletion Gate

`/4T/CXY/Neural_Gaffer_original_main_baseline` has already been removed. It was only a compatibility symlink to `external/official_neural_gaffer_baseline`.

Before deleting `/4T/CXY/Neural_Gaffer_original`, confirm that:

- No active scripts/configs outside docs, logs, effects, or archived manifests reference `/4T/CXY/Neural_Gaffer_original`.
- No running render/preprocess/monitor processes still use `/4T/CXY/Neural_Gaffer_original` as a compatibility path; check with `pgrep -af /4T/CXY/Neural_Gaffer_original`.
- `logs/dataset_unions/*/meta.json` source paths point at `external_data/neural_gaffer_original`.
- A full dataset training smoke can open data from `configs/datasets/full_current_original_official2000_ecommerce1000_3dfuture_landscape_allready_plus_officialval.txt`.
- Same-batch manifest generation and paper panel generation succeed from the consolidated paths.

Do not delete the migrated `external_data/neural_gaffer_original/` tree unless the corresponding data has been backed up elsewhere.

## Cleanup Performed

After the migration smoke tests passed, the following redundant local outputs were removed from `Neural_Gaffer/`:

- `logs/smoke/`
- `logs/dataset_unions_legacy_pre_migration_20260410/`
- `logs/stage_flow_demo_20260329/`
- `logs/stage_flow_demo_large_20260329/`
- `logs/relighting_comparison/minimal_highlight_experiment_smoke_20260402/`
- duplicate checkpoint/export directories under `logs/neural_gaffer_res256/`, `logs/neural_gaffer_training0316/`, `logs/neural_gaffer_training_gpu1_highlight/`, and `logs/neural_gaffer_training_fullall_officialval/`
- smoke W&B offline run `wandb/offline-run-20260410_035450-h0hpubu5`
- redundant JSON copies of dataset quality/validation reports; the Markdown summaries were kept
- Python bytecode caches under source/script directories

The `change1` 20k checkpoint was moved, not discarded:

- `model_weights/neural_gaffer_model_cache/change1_ckpt20k__neural_gaffer_training_change1`

`/4T/CXY/Neural_Gaffer_original_main_baseline` was removed because it was only a compatibility symlink to `external/official_neural_gaffer_baseline`.

`/4T/CXY/Neural_Gaffer_original` was not removed because render/preprocess processes were still using its compatibility paths at cleanup time.

## Effects And Cache Cleanup

The paper/effects area was trimmed to keep all-method outputs and compact tabular summaries:

- removed `effects/tmp_local/`
- removed `logs/relighting_comparison/raw_env_cache/`
- removed `effects/0407/comparison_figures_clean_input_white_methods_gt_hdrbg_v2/panels/` and `panel_manifests/`; kept `all_methods_only/`
- removed duplicate `effects/0408/foreground_highlight_supervision_ablation_only_v1/grouped_panels*` and `scene_bg_assets/`; kept the fullwidth six-method preview and tables
- removed derived `all_sorted_*`, `not_ours_best`, and `ours_min_highlight_diff` panels under `effects/contrast/official_curated_highlight_hdri_v1/grouped_panels_v2/`; kept one `ours_best/` slice
- removed failure-only panels/manifests under official curated HDRI contrast outputs; kept `panels/all_samples/`
- removed stale JSON manifests that only pointed back to deleted `effects/tmp_local/` sources; kept Markdown/CSV summaries
- removed W&B per-run `tmp/` directories

One residual cache tree remains at `effects/contrast/official_curated_highlight_hdri_fullpool_v1/proxy/`. Deleting it entered a long filesystem wait on this machine, so it is safe to retry later when disk IO is quiet.
