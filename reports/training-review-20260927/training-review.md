# YOLO26s completed-run review

Recorded UTC: 2026-09-27T06:45:18.983192+00:00

All 300 epochs and the automatic evaluation of best.pt on the internal test set completed. No retraining or duplicate evaluation was performed. Original artifacts remain unchanged.

## Results

| Metric | Best validation (epoch 86) | Final validation (epoch 300) | Best checkpoint internal test |
|---|---:|---:|---:|
| Precision | 65.84% | 71.74% | 60.15% |
| Recall | 51.52% | 47.17% | 50.36% |
| mAP50 | 53.45% | 50.60% | 50.89% |
| mAP50-95 | 30.68% | 28.79% | 28.94% |

Best epoch is selected by validation mAP50-95 in results.csv. Test evaluation explicitly loaded best.pt. Precision/recall/F1 are confidence-curve summaries, not a fixed deployment operating point. The internal holdout is not the official unlabeled challenge benchmark.

## Per-class test results

| Class | Train boxes | Val boxes | Test boxes | Precision | Recall | F1 | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D00 longitudinal cracks | 16409 | 4632 | 1754 | 60.69% | 57.92% | 59.27% | 60.11% | 32.43% |
| D10 transverse cracks | 7320 | 1991 | 796 | 57.76% | 59.30% | 58.52% | 58.40% | 28.06% |
| D20 alligator cracks | 5750 | 1466 | 604 | 68.35% | 62.57% | 65.33% | 67.82% | 37.58% |
| D30 repaired cracks | 294 | 106 | 16 | 6.64% | 6.25% | 6.44% | 5.62% | 2.50% |
| D40 potholes | 2859 | 728 | 296 | 52.59% | 42.23% | 46.85% | 44.09% | 19.71% |
| D50 pedestrian crossing blurs | 652 | 159 | 59 | 65.95% | 69.49% | 67.68% | 73.25% | 45.99% |
| D60 lane line blurs | 3546 | 1027 | 420 | 55.46% | 58.10% | 56.75% | 54.99% | 31.40% |
| D70 manhole covers | 2717 | 698 | 296 | 77.66% | 84.55% | 80.96% | 87.37% | 49.12% |
| D80 patchy road sections | 647 | 201 | 57 | 56.35% | 63.16% | 59.56% | 57.22% | 42.60% |
| D90 rutting | 40 | 3 | 1 | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |

D90 has only ONE test box. Its displayed precision of 1.0 alongside zero recall and AP is not evidence of successful detection; it is a degenerate summary for this class. D30 has only 16 test boxes, so uncertainty is also substantial. Do not exclude these classes retrospectively to inflate the headline score.

## Configuration and timing

N-RDD2024 Mendeley DOI 10.17632/27c8pwsd6v.3; ten classes; 13,782 training, 3,704 validation, 1,509 internal-test images. Test contains 4,299 boxes. Original-ID and identical-pixel grouping was used, but neighboring road frames may remain correlated; no verified route-level separation.

YOLO26s: 9,955,604 parameters, 260 layers, 22.8 GFLOPs in the training summary; fused evaluation summary: 9,469,050 parameters, 120 layers, 20.8 GFLOPs. These different summaries represent different execution forms, not different experiments. Input640, batch4, AMP, MuSGD, seed42, early stopping disabled, mosaic closed for final10epochs. GPU RTX4060 Laptop8GB; Ultralytics8.4.161/PyTorch2.11.0+cu128. Full arguments and hashes are in statistics.json. The saved project argument retains the former mirror path, but actual save_dir and artifacts are in the Desktop project.

Sum of CSV elapsed counters across restart segments: 63.79 hours. This excludes interruption downtime and final evaluation, and is not pure GPU compute time. Segments: [{"first_epoch": 1, "last_epoch": 1, "reported_elapsed_seconds": 874.403}, {"first_epoch": 2, "last_epoch": 175, "reported_elapsed_seconds": 135830.0}, {"first_epoch": 176, "last_epoch": 300, "reported_elapsed_seconds": 92952.3}].

Reported test timing: 0.4ms preprocessing + 7.4ms inference + 2.2ms postprocessing per image. This is batched evaluation, not a demonstrated live-system FPS.

## What the evidence shows

- Validation mAP50-95 fell by 1.88 percentage points from its best epoch. Training classification loss fell from 1.711 to 0.584, while validation classification loss rose from 1.596 to 1.930. This is consistent with overfitting; 300 epochs did not improve the peak. Final augmentation changes also complicate training-loss comparisons.
- Test AP50 ranges from87.37% for manholes to5.62% for repaired cracks; potholes achieve44.09%. One overall score hides substantial class differences.
- The saved confusion matrix shows many unmatched ground truths and detections, with relatively fewer cross-class confusions. At its plotting thresholds, D00 has711 unmatched ground truths and523 unmatched predictions; D40 has168 and89. These are not counts at the confidence-curve optimum, and unmatched detections can include annotation/matching issues.
- Class support is extremely uneven: D00 has16,409 training boxes versus40 rutting and294 repaired-crack boxes. Scarcity is a plausible contributor, not a proven sole cause.
- Bounding-box localisation, label consistency and small-object resolution are hypotheses needing validation. No systematic annotation audit or size-stratified analysis has yet established their contribution.

## Models with relevant published evidence

| Model/source | Reported results | Comparability and decision |
|---|---|---|
| YOLOv9s, YOLO-RD paper Table11 |46.9% AP50;26.9% AP50-95 |Strong baseline within that paper; seven-class cleaned N-RDD2024 split, not ours. First alternative to reproduce. |
| YOLO-RD, same Table11 |45.5%;26.2% |Custom attention and balanced loss; useful source of class-imbalance ideas. Requires verified implementation. |
| Modified RT-DETR-R18, Applied Sciences16(14)7177 |72.4%;44.4% |Promising lead, not a verified equivalent benchmark. See qualification below. |
| YOLOv8, Infrastructures11(6)207 |84.6% accuracy;89.6% precision;86.3% recall |Four-class task; accuracy must not be relabelled mAP. Architecture scale and split need clarification before reproduction. |

YOLO-RD uses12,625train/1,750val/3,492test after cleaning and selects seven damage categories; its test is derived from official validation. Our ten-class scores cannot rank above or below its models fairly. Source: local Downloads/YOLO.pdf, p17 Table11 and section4.6.3; https://doi.org/10.1016/j.neucom.2025.132000 .

The RT-DETR paper calls its dataset SVRDD2024 but links an N-RDD2024 repository and lists the same ten labels. It says seven classes are selected while listing four excluded labels, leaving six. Its baseline prose also alternates AP50 figures (64.7 versus70.2). Without exact split manifests, class mapping and code, the72.4% score is an unresolved lead. Confirm reproducibility before investing in this custom model. Publisher direct retrieval returned429; indexed publisher passages in sections4.1/4.4 were inspected. https://doi.org/10.3390/app16147177 .

The YOLOv8 pavement-condition paper uses four categories and reports20,620 images, versus our18,995 annotated-image version. Its high reported accuracy is not verified as AP50; do not treat it as evidence of an84.6% ten-class benchmark. It is a reproduction lead, not proof that changing to YOLOv8 will yield that score. https://doi.org/10.3390/infrastructures11060207 .

The earlier Springer lightweight-YOLOv8 and Robust multi-expert papers use other dataset setups; they motivate techniques but are not established N-RDD2024 scores. No verified superior model on our exact ten-class v3 split has been found.

## Ranked improvement experiments (proposed, not run)

1. Keep best.pt as the frozen baseline. Audit training/validation D30 and D90 labels and confusing examples; collect more independent annotations for rare classes. Preserve existing splits and create a versioned dataset if changing labels or split design. Do not use test examples for tuning.
2. Reproduce YOLOv9s and YOLOv8s on our identical ten-class split at640, with comparable pretraining/compute and early stopping (initial patience30-50). Compare validation AP per class and AP50-95, runtime and peak memory. Repeat promising comparisons with three seeds; epoch count alone is not compute parity.
3. Test640 versus832 input or a carefully defined tiling approach on validation, holding other settings fixed. Hypothesis: improved small-crack localisation; measure recall by size and latency. RTX4060 memory may require a smaller batch with matched effective batch.
4. Test capped training-only rare-class sampling versus baseline; separately test a documented class-balanced loss if compatible. Include controls because naive duplication may overfit40 rutting boxes and increase false positives.
5. Test a small, predeclared learning-rate/augmentation schedule comparison; use early stopping and best validation checkpoint. Do not spend another300epochs without monitoring generalisation.
6. Investigate RT-DETR-R18 only after resolving dataset/code discrepancies and confirming single-GPU feasibility. Treat its published score as a lead, not a promised improvement.

Choose settings on validation. The present test set has now been inspected; reserve a fresh route-separated external test for final claims and local deployment. Threshold tuning can change precision/recall at deployment but cannot by itself prove better AP. New models, training and test-driven parameter searches were not launched.

## Saved evidence

Original run: ${PROJECT_ROOT}\runs\nrdd2024_yolo26s_300

Test metrics: ${PROJECT_ROOT}\runs\evaluation\test\metrics.json

Confusion matrix: ${PROJECT_ROOT}\runs\evaluation\test\confusion_matrix.png

Machine-readable report includes per-class support, full settings, checkpoint/file SHA256 hashes, and restart-aware timing.
