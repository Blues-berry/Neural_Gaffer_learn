# Ablation Reference

本文件记录当前高光监督、频域辅助、hybrid probe 等消融的配置和出图方式。

## 方法配置

- 主线: `configs/methods/highlight_full_main_showcase.txt`
- image-space highlight 低学习率: `configs/methods/highlight_spuru_cosine_lowlr.txt`
- hybrid probe: `configs/methods/highlight_hybrid_probe.txt`
- frequency separation auxiliary: `configs/methods/highlight_frequency_auxiliary_spuru.txt`

## Ablation Checkpoint Roots

- baseline 0316: `model_weights/neural_gaffer_model_cache/neural_gaffer_training0316`
- ours 80k: `model_weights/neural_gaffer_model_cache/jbhdfvfc_ckpt80k__neural_gaffer_training_gpu1_highlight`
- hyblite: `model_weights/neural_gaffer_model_cache/hyblite__imgsoft-hyb30to50k-his60-hps75-q88-gb10-r40-w2-g2-lp0.03-wu3k-lpwu5k-80k-spuru_hyblite_a-0331-02`
- freqsplit: `model_weights/neural_gaffer_model_cache/freqsplit__imgsoft-q88-gb10-r40-w2-g2-freq-fl8-fh3-fs15-wu3k-80k-spuru_freqsplit_a-0402-01`
- change1 20k: `model_weights/neural_gaffer_model_cache/change1_ckpt20k__neural_gaffer_training_change1`

## 推荐 Suite

- `configs/comparison_suites/validation_samebatch_0407_ablations_source.json`
- `configs/comparison_suites/foreground_highlight_realchain_substitutes_0409.json`
- `configs/comparison_suites/foreground_highlight_realchain_substitutes_fullwidth_0409.json`

## 运行

```bash
cd /4T/CXY/Neural_Gaffer
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/python scripts/run_checkpoint_panel_suite.py \
  --manifest logs/samebatch/run_YYYYMMDD/manifests/same_batch_manifest_native512.json \
  --suite configs/comparison_suites/validation_samebatch_0407_ablations_source.json \
  --pred-root logs/ablation/run_YYYYMMDD/predictions \
  --assets-dir logs/ablation/run_YYYYMMDD/assets \
  --panel-output logs/ablation/run_YYYYMMDD/panels/ablation_panel.png \
  --device cuda:1 \
  --resolution 256 \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --skip-existing
```

## 论文表格

当前论文相关结果集中在:

- `effects/0408/foreground_highlight_supervision_ablation_only_v1/`
- `effects/0408/foreground_highlight_supervision_ablation_only_v1/tables/`

这些目录属于论文产物，不纳入日志清理。
