# Internal Clean Ablation Module Table

| method | ImgSpace | ImgSpace W | HLW | Quantile | Q | Min | Max | Blur | Relative | Kernel | FG | Rand |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Abl. Base | off | 0.1000 | 2.0000 | off | 0.8800 | 0.0200 | 0.2000 | 0.0000 | none | 15 | 0.9800 | 0.4000 |
| Abl. ImgSpace Fixed | on | 0.1000 | 2.0000 | off | 0.8800 | 0.0200 | 0.2000 | 0.0000 | none | 15 | 0.9800 | 0.4000 |
| Abl. Quantile | on | 0.1000 | 2.0000 | on | 0.8800 | 0.6000 | 0.9500 | 0.0000 | none | 15 | 0.9800 | 0.4000 |
| Abl. Blur | on | 0.1000 | 2.0000 | on | 0.8800 | 0.6000 | 0.9500 | 1.0000 | none | 15 | 0.9800 | 0.4000 |
| Abl. Relative | on | 0.1000 | 2.0000 | on | 0.8800 | 0.0200 | 0.2000 | 1.0000 | difference | 15 | 0.9800 | 0.4000 |
| Abl. Full Main | on | 0.1000 | 2.0000 | on | 0.8800 | 0.0200 | 0.2000 | 1.0000 | difference | 15 | 0.9600 | 0.4000 |
