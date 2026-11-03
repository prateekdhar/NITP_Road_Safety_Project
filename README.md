# N-RDD2024 Road Damage Detection and Robustness Study

This repository contains the code and reports for a Windows/RTX 4060 road-damage detection project using N-RDD2024 version 3 (10 classes). It includes baseline training, evaluation and inference scripts, noise/blur training utilities, model-comparison reports, the completed 14-run YOLO26s ablation study, and a draft LaTeX review paper.

## Key results

The completed single-seed 100-epoch study found that YOLO26s at 800-pixel input had the strongest validation result among the tested runs: precision 69.18%, recall 50.49%, derived F1 58.37%, mAP@50 54.59%, and mAP@50–95 31.36%. The 640-pixel YOLO26s control scored 68.82%, 49.66%, 57.69%, 54.02%, and 31.17%, respectively. The higher-resolution run costs more computation. Results and limitations are documented in [`reports/ablation-study-20261103/README.md`](reports/ablation-study-20261103/README.md). These are validation metrics, not held-out test accuracy; each variant used one seed.

Earlier model comparisons are in [`reports/model-report-100epochs-20261019/`](reports/model-report-100epochs-20261019/). The review manuscript draft is in [`reports/review-paper-20261016/`](reports/review-paper-20261016/).

## Dataset and checkpoints

Dataset information and the version-3 Mendeley file manifest are in [`data/`](data/). Dataset images/labels and pretrained/trained model weights are not stored in Git. Download N-RDD2024 version 3 from [Mendeley Data](https://doi.org/10.17632/27c8pwsd6v.3), subject to its license and terms, then run `download_dataset.py` and `prepare_dataset.py`. The training scripts expect the local paths described in their configuration or command-line help. Ultralytics checkpoints are downloaded by Ultralytics or supplied locally; review the [Ultralytics license](https://www.ultralytics.com/license) before redistribution or commercial use.

## Setup

Use Python 3.12 and a CUDA-enabled PyTorch build compatible with your NVIDIA driver. On Windows, open PowerShell in this folder and follow `setup.ps1`, then prepare the dataset before training. `requirements-lock.txt` records the original environment and may contain platform-specific package pins.

For the 100-epoch ablation study, see `run_yolo26_ablation.py` and `ablation_components.py`. This study has already completed; rerunning the queue writes new experiment directories only when configured to do so. Do not run it on top of the original local experiment folder without checking the existing-run safeguards.

## Published project contents

The public repository contains source code, documentation, manuscript outputs and metrics/provenance needed to inspect the work. It excludes the 2.6 GB dataset, virtual environment, bulky checkpoints, training logs, validation image dumps, downloaded journal PDFs, and private machine-specific cache files. The Git history contains prior repository commits; this update replaces the current `main` tree but does not rewrite historical commits.
