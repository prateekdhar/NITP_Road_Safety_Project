# YOLO26s ablation and training-setting study
Completed 3 November 2026. N-RDD2024 v3, ten classes, fixed seed 42, 100 epochs per run. All numbers below are validation results on the original clean validation split. The held-out internal test split was not used to select these experiments.
F1 is derived from reported mean precision and recall; it is not an independently averaged per-class F1. Each run used one seed. Small differences should not be interpreted as statistically significant.
## Results ranked by mAP@50–95
| Run | Best epoch | Input | Precision | Recall | F1* | mAP@50 | mAP@50–95 | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resolution_800 | 61 | 800 | 69.18% | 50.49% | 58.37% | 54.59% | 31.36% | 35.80704 |
| close_mosaic_20 | 53 | 640 | 68.82% | 49.66% | 57.69% | 54.02% | 31.17% | 22.7782656 |
| control | 53 | 640 | 68.82% | 49.66% | 57.69% | 54.02% | 31.17% | 22.7782656 |
| noise_010 | 66 | 640 | 69.55% | 50.05% | 58.21% | 53.74% | 30.94% | 22.7782656 |
| p2 | 64 | 640 | 68.84% | 49.12% | 57.33% | 53.77% | 30.92% | 24.976384 |
| lr_double | 65 | 640 | 68.53% | 49.67% | 57.60% | 53.98% | 30.83% | 22.7782656 |
| wd_high | 63 | 640 | 69.96% | 48.40% | 57.22% | 53.47% | 30.77% | 22.7782656 |
| wd_low | 53 | 640 | 68.61% | 50.31% | 58.05% | 53.62% | 30.77% | 22.7782656 |
| p2_fusion | 52 | 640 | 67.09% | 49.31% | 56.84% | 53.32% | 30.74% | 24.976384 |
| fusion | 53 | 640 | 69.73% | 49.04% | 57.58% | 53.77% | 30.70% | 22.7782656 |
| noise_025 | 67 | 640 | 69.65% | 50.05% | 58.25% | 53.35% | 30.60% | 22.7782656 |
| lr_half | 62 | 640 | 69.83% | 49.01% | 57.59% | 53.19% | 30.50% | 22.7782656 |
| mosaic_050 | 63 | 640 | 68.89% | 49.11% | 57.34% | 52.95% | 29.93% | 22.7782656 |
| mosaic_025 | 48 | 640 | 67.29% | 48.57% | 56.42% | 51.83% | 29.30% | 22.7782656 |

## Finding
The 800-pixel YOLO26s run had the highest mAP@50–95 (31.36%) and improved all five reported metrics over the 640-pixel control (31.17% mAP@50–95). It also used 35.81 GFLOPs versus 22.78 GFLOPs. Resolution and inference compute changed together; this is a resolution/compute trade-off, not an isolated architecture effect.
The P2, weighted-fusion, and combined variants did not exceed the control on mAP@50–95. Noise augmentation and training-setting variants did not show a consistent gain across all metrics. The study is an initial single-seed screen; promising findings need repeated seeds and, after choices are locked, a final evaluation on the internal test set.
The 10% and 25% noise/blur variants apply synthetic Gaussian sensor noise or blur during training. They do not establish robustness to real dashcam artifacts.
## Files
Each run folder includes its per-epoch `results.csv`, training `args.yaml`, provenance, and final validation record. Checkpoint binaries and validation image dumps are not included.
*F1 is derived from mean precision and recall.
