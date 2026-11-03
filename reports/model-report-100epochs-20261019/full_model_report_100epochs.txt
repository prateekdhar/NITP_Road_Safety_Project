# Road-damage detection: complete run report, capped at 100 epochs

Prepared 19 October 2026 (Asia/Calcutta) from local saved artifacts. No training or evaluation was launched.

## Scope and comparison rules

The main table uses the best recorded training-validation epoch among epochs 1–100, selected by mAP50–95. All metrics in this table come from results.csv, avoiding mixing CSV values with later re-evaluations. Latest means the final epoch within the cap. Scores are percentages. F1* is derived from mean precision and mean recall; it is not macro-averaged per-class F1 or a fixed-confidence deployment metric.

Three 30-epoch fine-tunes are reported separately: they start from the noise-augmented YOLOv9s checkpoint selected from a 100-epoch run. Their local epoch counters do not represent total training exposure. They are not eligible as fresh models trained within a strict total budget of 100 epochs. Smoke/preflight runs are excluded from research rankings and listed in the appendix.

## Dataset

N-RDD2024, Mendeley 27c8pwsd6v version 3; 10 classes (D00 through D90); 13,782 train / 3,704 validation / 1,509 internal-test images. These are project splits, not the seven-class split used in YOLO-RD. D90 has only 40/3/1 train/validation/test boxes; its class metrics are highly unstable. Validation results are not held-out test accuracy.

## Best validation results within the first 100 epochs

| Model | Epochs included | Best epoch | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv9s noise-augmented | 100 | 98 | 66.04 | 50.64 | 57.32 | 53.26 | 30.94 |
| YOLO26s | 100 | 86 | 65.84 | 51.52 | 57.80 | 53.45 | 30.68 |
| YOLOv9s baseline | 100 | 97 | 66.27 | 49.98 | 56.98 | 52.57 | 30.47 |
| YOLO12s | 100 | 80 | 65.79 | 49.18 | 56.29 | 52.51 | 30.21 |
| YOLOv10s | 100 | 78 | 65.68 | 48.89 | 56.06 | 52.25 | 29.89 |
| YOLO11s | 100 | 74 | 68.17 | 47.26 | 55.82 | 51.75 | 29.66 |
| YOLOv8s | 100 | 80 | 68.09 | 46.89 | 55.53 | 50.95 | 29.32 |
| YOLOv5su | 100 | 71 | 65.48 | 46.84 | 54.61 | 50.65 | 29.05 |
| YOLOv6s (scratch) | 100 | 100 | 68.13 | 44.66 | 53.96 | 48.23 | 27.66 |

## Supplement: fine-tunes with prior training exposure

| Model | Epochs included | Best epoch | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv9s augmentation fine-tune | 30 | 3 | 64.77 | 48.31 | 55.34 | 50.42 | 28.92 |
| YOLOv9s consistency fine-tune | 30 | 25 | 65.02 | 49.04 | 55.91 | 50.49 | 28.69 |
| YOLOv9s proposed fine-tune | 30 | 3 | 64.03 | 48.23 | 55.01 | 50.11 | 28.52 |

## YOLOv9s noise-augmented

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 98.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 66.036 | 50.637 | 57.320 | 53.260 | 30.939 |
| Last within cap | 65.789 | 50.339 | 57.036 | 53.107 | 30.934 |

Architecture: parameters 7291278; GFLOPs at 640: 27.4. Best checkpoint size: 15.25 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 97901.1 seconds; detected time-counter resets: 1. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\runs\nrdd2024_yolov9s_noise_100_oct04\weights\last.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 2
project: ${PROJECT_ROOT}\runs
name: nrdd2024_yolov9s_noise_100_oct04
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: ${PROJECT_ROOT}\runs\nrdd2024_yolov9s_noise_100_oct04\weights\last.pt
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\runs\nrdd2024_yolov9s_noise_100_oct04

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 65.905 | 50.546 | 57.213 | 53.129 | 30.915 |
| sensor_noise | 60.932 | 34.559 | 44.104 | 36.482 | 20.574 |
| motion_diagonal | 43.469 | 17.704 | 25.160 | 15.353 | 7.830 |
| defocus | 50.374 | 28.369 | 36.297 | 27.965 | 14.593 |
| mixed_diagonal | 38.851 | 12.969 | 19.447 | 9.344 | 4.470 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 64.444 | 53.735 | 58.501 | 33.105 |
| D10 | 61.124 | 54.897 | 58.572 | 29.080 |
| D20 | 66.394 | 60.232 | 64.299 | 35.738 |
| D30 | 44.043 | 33.962 | 33.238 | 16.583 |
| D40 | 63.294 | 47.609 | 51.940 | 24.345 |
| D50 | 63.956 | 65.409 | 67.284 | 43.438 |
| D60 | 53.939 | 50.633 | 49.462 | 28.155 |
| D70 | 78.201 | 77.794 | 82.183 | 46.350 |
| D80 | 63.657 | 61.194 | 65.214 | 51.933 |
| D90 | 100.000 | 0.000 | 0.598 | 0.419 |

Saved speed (milliseconds/image): {"preprocess": 0.36544127967512235, "inference": 11.419901295894169, "loss": 0.0032930075492120002, "postprocess": 1.073796544217895}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\runs\nrdd2024_yolov9s_noise_100_oct04\results.csv`

Checkpoint: `${PROJECT_ROOT}\runs\nrdd2024_yolov9s_noise_100_oct04\weights\best.pt`

Checkpoint SHA-256: `18c5796b9f03c3440ec5f3fabd19bf2e63a1fd53e12612b1e087c03a96c3069b`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261012_130732\comparison.json`

## YOLO26s

Recorded epochs: 300; included local epochs: 100. Best within cap: epoch 86.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 65.836 | 51.518 | 57.804 | 53.446 | 30.678 |
| Last within cap | 65.466 | 50.867 | 57.250 | 53.235 | 30.513 |

Architecture: parameters 9955604; GFLOPs at 640: 22.8. Best checkpoint size: 20.37 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 74977.2 seconds; detected time-counter resets: 1. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\runs\nrdd2024_yolo26s_300\weights\last.pt
data: ${PROJECT_ROOT}\data\processed\dataset.yaml
epochs: 300
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 2
project: ${PROJECT_ROOT}\runs
name: nrdd2024_yolo26s_300
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: ${PROJECT_ROOT}\runs\nrdd2024_yolo26s_300\weights\last.pt
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\runs\nrdd2024_yolo26s_300

```

Per-class results and inference speed are not included where an eligible, hash-verified evaluation record was not located. No values are inferred.

### Provenance

Results: `${PROJECT_ROOT}\runs\nrdd2024_yolo26s_300\results.csv`

Checkpoint: `${PROJECT_ROOT}\runs\nrdd2024_yolo26s_300\weights\best.pt`

Checkpoint SHA-256: `3824502cba92efaf6c7d4e859af715bcadd85efda1af952da66139013ba6a83f`

## YOLOv9s baseline

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 97.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 66.271 | 49.979 | 56.983 | 52.571 | 30.470 |
| Last within cap | 66.284 | 49.934 | 56.959 | 52.582 | 30.385 |

Architecture: parameters 7291278; GFLOPs at 640: 27.4. Best checkpoint size: 15.25 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 195207.0 seconds; detected time-counter resets: 0. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\weights\yolov9s.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 2
project: ${PROJECT_ROOT}\runs
name: nrdd2024_yolov9s_baseline_100_oct04
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\runs\nrdd2024_yolov9s_baseline_100_oct04

```

### Saved best-checkpoint corruption evaluation

Split: val; batch see archived record; corruption seed see archived record. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean_0 | 66.032 | 50.213 | 57.046 | 52.566 | 30.457 |
| sensor_noise_0.03 | 63.590 | 46.664 | 53.827 | 48.100 | 27.459 |
| sensor_noise_0.06 | 59.095 | 38.541 | 46.655 | 38.998 | 21.992 |
| lowlight_blur_0.6 | 64.867 | 46.857 | 54.410 | 49.379 | 28.222 |
| lowlight_blur_0.35 | 64.012 | 41.588 | 50.420 | 44.608 | 25.254 |
| clean | 66.032 | 50.213 | 57.046 | 52.566 | 30.457 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 64.627 | 53.998 | 58.500 | 33.100 |
| D10 | 61.110 | 56.036 | 58.346 | 28.903 |
| D20 | 66.548 | 58.081 | 63.277 | 35.007 |
| D30 | 51.732 | 34.906 | 34.946 | 18.112 |
| D40 | 62.529 | 47.940 | 51.074 | 23.457 |
| D50 | 64.002 | 64.780 | 65.458 | 41.152 |
| D60 | 53.730 | 50.049 | 49.013 | 27.867 |
| D70 | 77.925 | 77.650 | 81.950 | 46.447 |
| D80 | 58.122 | 58.692 | 62.798 | 50.315 |
| D90 | 100.000 | 0.000 | 0.296 | 0.208 |

Saved speed (milliseconds/image): {}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\runs\nrdd2024_yolov9s_baseline_100_oct04\results.csv`

Checkpoint: `${PROJECT_ROOT}\runs\nrdd2024_yolov9s_baseline_100_oct04\weights\best.pt`

Checkpoint SHA-256: `4e3731971fc26b01efca24cf69efd6a641a3c0a80d6b34a4d9303a8a0f0c1569`

Evaluation: `${PROJECT_ROOT}\reports\completed-study-20261010\statistics.json`

## YOLO12s

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 80.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 65.787 | 49.183 | 56.286 | 52.510 | 30.214 |
| Last within cap | 67.386 | 49.143 | 56.836 | 52.625 | 29.965 |

Architecture: parameters 9257006; GFLOPs at 640: 23.502592. Best checkpoint size: 18.95 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 53628.0 seconds; detected time-counter resets: 1. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo12s\weights\last.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 0
project: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs
name: yolo12s
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo12s\weights\last.pt
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo12s

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 65.757 | 49.169 | 56.266 | 52.513 | 30.192 |
| sensor_noise | 47.834 | 21.864 | 30.011 | 20.017 | 11.097 |
| motion_diagonal | 43.695 | 19.542 | 27.006 | 16.950 | 8.171 |
| defocus | 48.575 | 26.338 | 34.156 | 23.739 | 11.447 |
| mixed_diagonal | 2.162 | 5.643 | 3.126 | 1.731 | 0.842 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 67.543 | 52.763 | 59.512 | 33.813 |
| D10 | 61.382 | 54.495 | 57.738 | 27.818 |
| D20 | 67.142 | 59.277 | 64.630 | 35.792 |
| D30 | 40.624 | 26.415 | 31.874 | 16.542 |
| D40 | 62.281 | 46.841 | 51.699 | 23.417 |
| D50 | 63.158 | 64.690 | 64.158 | 40.666 |
| D60 | 55.598 | 50.355 | 48.867 | 28.047 |
| D70 | 79.251 | 77.650 | 82.510 | 45.954 |
| D80 | 60.592 | 59.204 | 64.142 | 49.873 |
| D90 | 100.000 | 0.000 | 0.000 | 0.000 |

Saved speed (milliseconds/image): {"preprocess": 0.34377904959034566, "inference": 7.77399244058043, "loss": 0.0027830455842936816, "postprocess": 1.0378659819849954}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo12s\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo12s\weights\best.pt`

Checkpoint SHA-256: `11be91c9b2f3f517f9fcfd49e2af637856c25a52a0f15b2c8be521bc3e75f50e`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261019_005911\comparison.json`

## YOLOv10s

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 78.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 65.678 | 48.892 | 56.055 | 52.252 | 29.889 |
| Last within cap | 66.221 | 49.491 | 56.647 | 52.040 | 29.638 |

Architecture: parameters 8074092; GFLOPs at 640: 24.9266688. Best checkpoint size: 16.55 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 52064.7 seconds; detected time-counter resets: 0. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\weights\yolov10s.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 4
project: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs
name: yolov10s
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov10s

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 66.030 | 48.764 | 56.099 | 52.249 | 29.890 |
| sensor_noise | 49.028 | 21.430 | 29.824 | 20.403 | 11.067 |
| motion_diagonal | 39.383 | 15.360 | 22.100 | 13.589 | 6.149 |
| defocus | 49.585 | 24.098 | 32.434 | 23.606 | 11.182 |
| mixed_diagonal | 6.119 | 2.684 | 3.731 | 0.619 | 0.317 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 66.229 | 52.541 | 58.534 | 32.898 |
| D10 | 61.252 | 53.240 | 56.625 | 27.594 |
| D20 | 67.701 | 59.482 | 63.649 | 34.612 |
| D30 | 49.083 | 35.849 | 36.766 | 19.405 |
| D40 | 64.277 | 46.220 | 51.135 | 23.700 |
| D50 | 60.459 | 60.377 | 63.909 | 41.360 |
| D60 | 55.428 | 48.004 | 48.255 | 27.558 |
| D70 | 77.233 | 75.215 | 81.592 | 44.820 |
| D80 | 58.642 | 56.716 | 61.837 | 46.913 |
| D90 | 100.000 | 0.000 | 0.182 | 0.036 |

Saved speed (milliseconds/image): {"preprocess": 0.3728078564773661, "inference": 4.973479805657346, "loss": 0.003553185873520591, "postprocess": 1.141646328279985}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov10s\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov10s\weights\best.pt`

Checkpoint SHA-256: `f4ac97abeb1ff14ff84dc5177b927f25ccd0cd0dc35e0ffe362bfc07b7460be2`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261016_072419\comparison.json`

## YOLO11s

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 74.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 68.170 | 47.260 | 55.821 | 51.755 | 29.656 |
| Last within cap | 66.345 | 48.547 | 56.067 | 51.413 | 29.468 |

Architecture: parameters 9431662; GFLOPs at 640: 21.6858112. Best checkpoint size: 19.20 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 89998.1 seconds; detected time-counter resets: 0. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\weights\yolo11s.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 4
project: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs
name: yolo11s
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo11s

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 68.187 | 47.296 | 55.852 | 51.781 | 29.680 |
| sensor_noise | 48.010 | 18.584 | 26.795 | 17.216 | 9.416 |
| motion_diagonal | 38.429 | 14.256 | 20.797 | 11.696 | 5.282 |
| defocus | 46.189 | 23.226 | 30.909 | 21.759 | 10.292 |
| mixed_diagonal | 3.151 | 1.734 | 2.237 | 0.440 | 0.223 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 68.224 | 51.770 | 58.487 | 33.017 |
| D10 | 62.172 | 50.932 | 56.380 | 27.564 |
| D20 | 69.718 | 56.753 | 62.980 | 34.103 |
| D30 | 49.429 | 31.132 | 33.564 | 17.267 |
| D40 | 68.393 | 43.693 | 51.103 | 23.242 |
| D50 | 63.748 | 57.510 | 61.319 | 39.859 |
| D60 | 56.990 | 48.393 | 49.093 | 27.662 |
| D70 | 79.968 | 76.065 | 81.735 | 45.681 |
| D80 | 63.229 | 56.716 | 63.146 | 48.409 |
| D90 | 100.000 | 0.000 | 0.000 | 0.000 |

Saved speed (milliseconds/image): {"preprocess": 2.133862634785835, "inference": 49.19392702506767, "loss": 0.027012688793080053, "postprocess": 9.57259881190041}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo11s\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolo11s\weights\best.pt`

Checkpoint SHA-256: `c9c026aa3630c185db4f101599e2d5c03e06a29deeb6ef41a00f21f63c18d338`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261017_084245\comparison.json`

## YOLOv8s

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 80.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 68.089 | 46.887 | 55.533 | 50.949 | 29.323 |
| Last within cap | 65.433 | 49.236 | 56.191 | 50.937 | 29.060 |

Architecture: parameters 11139470; GFLOPs at 640: 28.6614016. Best checkpoint size: 22.54 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 31043.3 seconds; detected time-counter resets: 0. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\weights\yolov8s.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 4
project: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs
name: yolov8s
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov8s

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 68.103 | 46.866 | 55.523 | 50.958 | 29.317 |
| sensor_noise | 43.689 | 19.456 | 26.922 | 17.095 | 9.228 |
| motion_diagonal | 38.584 | 14.074 | 20.624 | 12.002 | 5.373 |
| defocus | 46.341 | 23.398 | 31.095 | 21.433 | 10.179 |
| mixed_diagonal | 4.543 | 2.439 | 3.174 | 0.673 | 0.289 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 68.209 | 50.443 | 57.966 | 32.724 |
| D10 | 63.501 | 50.944 | 55.945 | 27.035 |
| D20 | 69.813 | 57.896 | 62.816 | 33.785 |
| D30 | 52.124 | 27.736 | 30.498 | 17.018 |
| D40 | 66.113 | 42.343 | 50.072 | 22.957 |
| D50 | 60.679 | 59.748 | 60.980 | 39.936 |
| D60 | 57.033 | 45.624 | 46.918 | 26.479 |
| D70 | 79.758 | 75.215 | 80.764 | 44.985 |
| D80 | 63.802 | 58.706 | 63.622 | 48.253 |
| D90 | 100.000 | 0.000 | 0.000 | 0.000 |

Saved speed (milliseconds/image): {"preprocess": 0.3726329912036656, "inference": 5.2286890659161145, "loss": 0.0036228132088400615, "postprocess": 1.2139359070880733}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov8s\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov8s\weights\best.pt`

Checkpoint SHA-256: `4aef1d224c4025c529e348fbe2c9d6f379af457a85d8686ecba9e122dd7c7840`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261015_124402\comparison.json`

## YOLOv5su

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 71.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 65.479 | 46.841 | 54.614 | 50.647 | 29.047 |
| Last within cap | 66.009 | 47.193 | 55.037 | 50.711 | 28.958 |

Architecture: parameters 9126062; GFLOPs at 640: 24.0599552. Best checkpoint size: 18.54 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 25405.0 seconds; detected time-counter resets: 1. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\speed_adjustment_20261014_122808\resume.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 4
project: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs
name: yolov5su
exist_ok: false
pretrained: true
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\speed_adjustment_20261014_122808\resume.pt
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov5su

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 65.457 | 46.834 | 54.601 | 50.646 | 29.045 |
| sensor_noise | 42.105 | 19.397 | 26.558 | 16.414 | 8.449 |
| motion_diagonal | 37.932 | 13.083 | 19.456 | 11.420 | 5.044 |
| defocus | 44.393 | 22.947 | 30.255 | 20.200 | 9.819 |
| mixed_diagonal | 24.243 | 1.399 | 2.646 | 0.569 | 0.315 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 65.576 | 52.229 | 57.407 | 31.974 |
| D10 | 58.659 | 51.525 | 54.753 | 26.774 |
| D20 | 66.993 | 57.872 | 62.840 | 34.286 |
| D30 | 47.929 | 18.868 | 27.773 | 15.837 |
| D40 | 63.499 | 45.604 | 50.582 | 22.828 |
| D50 | 63.158 | 61.006 | 61.909 | 38.849 |
| D60 | 53.996 | 47.030 | 47.382 | 27.040 |
| D70 | 77.946 | 75.501 | 79.990 | 44.941 |
| D80 | 56.814 | 58.706 | 60.471 | 46.580 |
| D90 | 100.000 | 0.000 | 3.350 | 1.340 |

Saved speed (milliseconds/image): {"preprocess": 0.3374676026135369, "inference": 4.311902240824447, "loss": 0.002785853161044081, "postprocess": 0.9824738660719788}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov5su\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov5su\weights\best.pt`

Checkpoint SHA-256: `10d9560e82db949c43baa2ab8702713ff8b0e212b6f61729483881661f5e5437`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261014_193458\comparison.json`

## YOLOv6s (scratch)

Recorded epochs: 100; included local epochs: 100. Best within cap: epoch 100.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 68.133 | 44.662 | 53.956 | 48.228 | 27.655 |
| Last within cap | 68.133 | 44.662 | 53.956 | 48.228 | 27.655 |

Architecture: parameters 16307790; GFLOPs at 640: 43.9053824. Best checkpoint size: 32.86 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 29781.2 seconds; detected time-counter resets: 0. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: yolov6s.yaml
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 100
time: null
patience: 0
batch: 4
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 4
project: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs
name: yolov6s_scratch
exist_ok: false
pretrained: false
cls_remap: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov6s_scratch

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 67.991 | 44.679 | 53.923 | 48.215 | 27.651 |
| sensor_noise | 43.890 | 18.363 | 25.893 | 16.594 | 9.191 |
| motion_diagonal | 28.725 | 13.230 | 18.116 | 11.069 | 5.028 |
| defocus | 31.462 | 21.628 | 25.634 | 19.099 | 9.327 |
| mixed_diagonal | 28.786 | 1.344 | 2.569 | 0.947 | 0.445 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 64.436 | 52.029 | 55.960 | 31.281 |
| D10 | 60.463 | 52.185 | 55.257 | 26.856 |
| D20 | 63.998 | 58.868 | 63.403 | 35.146 |
| D30 | 83.593 | 8.491 | 20.775 | 12.483 |
| D40 | 60.583 | 41.802 | 45.841 | 20.919 |
| D50 | 61.156 | 62.893 | 61.558 | 39.209 |
| D60 | 52.098 | 46.056 | 44.710 | 24.718 |
| D70 | 72.788 | 74.212 | 77.055 | 42.265 |
| D80 | 60.799 | 50.249 | 57.586 | 43.634 |
| D90 | 100.000 | 0.000 | 0.000 | 0.000 |

Saved speed (milliseconds/image): {"preprocess": 0.3529857722104509, "inference": 6.125735502069059, "loss": 0.004241873655889634, "postprocess": 1.2933220031600698}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov6s_scratch\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\detector_comparison_20261014\runs\yolov6s_scratch\weights\best.pt`

Checkpoint SHA-256: `7698a6617ec706385519cd6367cd5f05d1acebdd9b750408ea1bd88430184c12`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261015_035819\comparison.json`

## YOLOv9s augmentation fine-tune

Recorded epochs: 30; included local epochs: 30. Best within cap: epoch 3.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 64.770 | 48.307 | 55.340 | 50.424 | 28.915 |
| Last within cap | 65.922 | 47.743 | 55.379 | 50.371 | 28.164 |

Architecture: parameters 7291278; GFLOPs at 640: 27.4. Best checkpoint size: 15.23 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 15164.4 seconds; detected time-counter resets: 1. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_augmentation_30ep_20261010_184826\pause-20261011-003958\resume.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 30
time: null
patience: 0
batch: 2
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 0
project: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_augmentation_30ep_20261010_184826
name: train
exist_ok: false
pretrained: true
cls_remap: true
optimizer: AdamW
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 0
resume: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_augmentation_30ep_20261010_184826\pause-20261011-003958\resume.pt
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: false
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.0001
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 1
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 0
mixup: 0
cutmix: 0
copy_paste: 0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_augmentation_30ep_20261010_184826\train

```

### Modification configuration

```json

{
  "variant": "augmentation",
  "teacher": "C:\\Users\\prate\\OneDrive\\Desktop\\RoadDamageDetection\\runs\\nrdd2024_yolov9s_baseline_100_oct04\\weights\\best.pt",
  "curriculum_epochs": 10,
  "easy_fraction": 0.7,
  "kl_weight": 0.2,
  "feature_weight": 0.1,
  "structure_weight": 0.05,
  "min_teacher_iou": 0.3,
  "clean_probability": 0.3
}

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 64.826 | 48.294 | 55.352 | 50.422 | 28.900 |
| sensor_noise | 58.145 | 35.183 | 43.839 | 35.728 | 20.121 |
| motion_diagonal | 42.970 | 18.551 | 25.915 | 16.756 | 8.313 |
| defocus | 53.924 | 30.829 | 39.230 | 31.079 | 16.658 |
| mixed_diagonal | 34.067 | 13.983 | 19.828 | 11.239 | 5.671 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 64.807 | 50.389 | 54.974 | 30.465 |
| D10 | 59.322 | 50.979 | 54.560 | 26.536 |
| D20 | 63.664 | 55.048 | 58.992 | 31.549 |
| D30 | 33.195 | 42.190 | 31.312 | 16.142 |
| D40 | 61.279 | 44.505 | 49.820 | 23.494 |
| D50 | 68.270 | 61.635 | 63.705 | 40.235 |
| D60 | 58.109 | 44.985 | 47.384 | 26.392 |
| D70 | 80.221 | 74.499 | 81.210 | 44.810 |
| D80 | 59.393 | 58.706 | 62.260 | 49.379 |
| D90 | 100.000 | 0.000 | 0.000 | 0.000 |

Saved speed (milliseconds/image): {"preprocess": 0.3559326943295221, "inference": 10.555717548720107, "loss": 0.0031756210716926045, "postprocess": 0.9986230832128962}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_augmentation_30ep_20261010_184826\train\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_augmentation_30ep_20261010_184826\train\weights\best.pt`

Checkpoint SHA-256: `0000647c8af1fdf5efd4c68ae24075023115856cd7abd1acfdbe7e694d164318`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261012_130732\comparison.json`

## YOLOv9s consistency fine-tune

Recorded epochs: 30; included local epochs: 30. Best within cap: epoch 25.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 65.016 | 49.042 | 55.910 | 50.488 | 28.687 |
| Last within cap | 56.243 | 47.451 | 51.474 | 49.899 | 28.175 |

Architecture: parameters 7291278; GFLOPs at 640: 27.4. Best checkpoint size: 15.23 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 11617.3 seconds; detected time-counter resets: 1. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_consistency_30ep_20261010_184826\pause-20261011-173427\resume.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 30
time: null
patience: 0
batch: 2
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 0
project: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_consistency_30ep_20261010_184826
name: train
exist_ok: false
pretrained: true
cls_remap: true
optimizer: AdamW
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 0
resume: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_consistency_30ep_20261010_184826\pause-20261011-173427\resume.pt
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: false
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.0001
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 1
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 0
mixup: 0
cutmix: 0
copy_paste: 0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_consistency_30ep_20261010_184826\train

```

### Modification configuration

```json

{
  "variant": "consistency",
  "teacher": "C:\\Users\\prate\\OneDrive\\Desktop\\RoadDamageDetection\\runs\\nrdd2024_yolov9s_baseline_100_oct04\\weights\\best.pt",
  "curriculum_epochs": 10,
  "easy_fraction": 0.7,
  "kl_weight": 0.2,
  "feature_weight": 0.1,
  "structure_weight": 0.05,
  "min_teacher_iou": 0.3,
  "clean_probability": 0.3
}

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 65.022 | 49.044 | 55.914 | 50.462 | 28.688 |
| sensor_noise | 50.056 | 37.289 | 42.739 | 38.157 | 21.401 |
| motion_diagonal | 42.644 | 18.030 | 25.344 | 16.405 | 8.255 |
| defocus | 45.878 | 31.216 | 37.153 | 32.797 | 17.762 |
| mixed_diagonal | 44.624 | 16.084 | 23.645 | 13.878 | 7.355 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 66.173 | 49.698 | 55.657 | 31.154 |
| D10 | 59.686 | 52.034 | 54.961 | 26.965 |
| D20 | 65.490 | 55.662 | 59.141 | 31.289 |
| D30 | 30.843 | 44.340 | 27.719 | 13.853 |
| D40 | 67.457 | 42.995 | 49.640 | 22.405 |
| D50 | 64.710 | 62.893 | 66.315 | 41.808 |
| D60 | 53.858 | 48.685 | 48.009 | 26.682 |
| D70 | 82.064 | 74.928 | 80.452 | 44.607 |
| D80 | 59.943 | 59.204 | 62.577 | 48.050 |
| D90 | 100.000 | 0.000 | 0.155 | 0.062 |

Saved speed (milliseconds/image): {"preprocess": 0.3387000539769596, "inference": 10.018020680395464, "loss": 0.0033220302048111616, "postprocess": 0.9988687095591172}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_consistency_30ep_20261010_184826\train\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_consistency_30ep_20261010_184826\train\weights\best.pt`

Checkpoint SHA-256: `6b24013826b3e8880d0e0dd63105572ee81ca284d92b83cbaf69d540e201dff6`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261012_130732\comparison.json`

## YOLOv9s proposed fine-tune

Recorded epochs: 30; included local epochs: 30. Best within cap: epoch 3.

| Selection | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| Best within cap | 64.030 | 48.225 | 55.015 | 50.114 | 28.516 |
| Last within cap | 64.283 | 48.228 | 55.110 | 49.738 | 27.878 |

Architecture: parameters 7291278; GFLOPs at 640: 27.4. Best checkpoint size: 15.23 decimal MB. YOLOv9s architecture counts are from the archived completed-study report; YOLO26s count requires the archived architecture record (see machine-readable sources).

CSV time at last included epoch: 50182.2 seconds; detected time-counter resets: 0. This is the logged counter, not total wall time across pauses/restarts. No total elapsed-time claim is made.

### Training configuration

```yaml

task: detect
mode: train
model: ${PROJECT_ROOT}\runs\nrdd2024_yolov9s_noise_100_oct04\weights\best.pt
data: ${PROJECT_ROOT}\experiments\oct04_200\baseline.yaml
epochs: 30
time: null
patience: 0
batch: 2
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 0
project: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_proposed_30ep_20261010_184826
name: train
exist_ok: false
pretrained: true
cls_remap: true
optimizer: AdamW
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 0
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0
compile: false
channels_last: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: false
nms: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
lr0: 0.0001
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 1
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 0
mixup: 0
cutmix: 0
copy_paste: 0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: ${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_proposed_30ep_20261010_184826\train

```

### Modification configuration

```json

{
  "variant": "proposed",
  "teacher": "C:\\Users\\prate\\OneDrive\\Desktop\\RoadDamageDetection\\runs\\nrdd2024_yolov9s_baseline_100_oct04\\weights\\best.pt",
  "curriculum_epochs": 10,
  "easy_fraction": 0.7,
  "kl_weight": 0.2,
  "feature_weight": 0.1,
  "structure_weight": 0.05,
  "min_teacher_iou": 0.3,
  "clean_probability": 0.3
}

```

### Saved best-checkpoint corruption evaluation

Split: val; batch 2; corruption seed 20261010. SHA-256 matches the current checkpoint. These are a separate evaluation pass; small numerical differences from CSV are expected.

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean | 64.127 | 48.271 | 55.081 | 50.137 | 28.539 |
| sensor_noise | 58.889 | 34.675 | 43.649 | 35.498 | 20.056 |
| motion_diagonal | 41.665 | 15.043 | 22.105 | 13.205 | 6.467 |
| defocus | 53.524 | 30.924 | 39.200 | 30.927 | 16.593 |
| mixed_diagonal | 35.774 | 14.591 | 20.728 | 11.889 | 6.110 |

### Per-class clean validation

| Class | Precision % | Recall % | AP50 % | AP50–95 % |
| --- | --- | --- | --- | --- |
| D00 | 62.763 | 50.762 | 55.062 | 30.680 |
| D10 | 59.450 | 50.377 | 54.026 | 26.459 |
| D20 | 62.826 | 57.026 | 59.253 | 31.415 |
| D30 | 34.461 | 43.653 | 33.176 | 17.062 |
| D40 | 62.541 | 42.995 | 48.869 | 22.770 |
| D50 | 65.683 | 59.748 | 62.128 | 38.328 |
| D60 | 60.674 | 46.120 | 47.011 | 25.996 |
| D70 | 77.341 | 74.819 | 80.976 | 44.856 |
| D80 | 55.529 | 57.214 | 60.871 | 47.825 |
| D90 | 100.000 | 0.000 | 0.000 | 0.000 |

Saved speed (milliseconds/image): {"preprocess": 0.35018342332988645, "inference": 10.350710745087566, "loss": 0.0032286447306670504, "postprocess": 1.016690955616537}. Hardware load, batch size, fusion and evaluation settings affect latency; this is not an end-to-end camera FPS benchmark.

### Provenance

Results: `${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_proposed_30ep_20261010_184826\train\results.csv`

Checkpoint: `${PROJECT_ROOT}\experiments\paper_noise_20261010\paper_proposed_30ep_20261010_184826\train\weights\best.pt`

Checkpoint SHA-256: `ec55450198d668afacf5066dbf93deabeb51d918db74c0b964e00b094e455dd2`

Evaluation: `${PROJECT_ROOT}\experiments\paper_noise_20261010\evaluation_20261012_130732\comparison.json`

## Validation class support

| Class | Images containing class | Boxes |
| --- | --- | --- |
| D00 | 2201 | 4632 |
| D10 | 1244 | 1991 |
| D20 | 1135 | 1466 |
| D30 | 58 | 106 |
| D40 | 438 | 728 |
| D50 | 142 | 159 |
| D60 | 728 | 1027 |
| D70 | 529 | 698 |
| D80 | 157 | 201 |
| D90 | 2 | 3 |

## Historical YOLOv9s robustness evaluation

This older corruption protocol is separate from the later paper-study protocol. Do not compare conditions by name alone.

baseline

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean_0 | 66.032 | 50.213 | 57.046 | 52.566 | 30.457 |
| sensor_noise_0.03 | 63.590 | 46.664 | 53.827 | 48.100 | 27.459 |
| sensor_noise_0.06 | 59.095 | 38.541 | 46.655 | 38.998 | 21.992 |
| lowlight_blur_0.6 | 64.867 | 46.857 | 54.410 | 49.379 | 28.222 |
| lowlight_blur_0.35 | 64.012 | 41.588 | 50.420 | 44.608 | 25.254 |

noise_robust

| Condition | Precision % | Recall % | F1* % | mAP50 % | mAP50–95 % |
| --- | --- | --- | --- | --- | --- |
| clean_0 | 65.904 | 50.541 | 57.209 | 53.127 | 30.907 |
| sensor_noise_0.03 | 66.784 | 49.204 | 56.662 | 52.661 | 30.368 |
| sensor_noise_0.06 | 65.733 | 47.005 | 54.813 | 50.181 | 28.970 |
| lowlight_blur_0.6 | 69.012 | 47.681 | 56.397 | 53.230 | 30.723 |
| lowlight_blur_0.35 | 68.223 | 47.461 | 55.979 | 52.748 | 30.379 |

## Methods note

YOLO26s used a 300-epoch training schedule; only its first 100 recorded epochs are included in this report.

## Interpretation and limitations

The main table is sorted by mAP50–95. Model generation alone does not predict performance. YOLOv6s was trained from scratch, YOLOv5su is the updated Ultralytics variant rather than original YOLOv5s, and the YOLOv6s implementation is the bundled Ultralytics version. Initialization, optimizer, learning-rate schedule and augmentation differ; these runs do not isolate architecture alone.

Each experiment has one seed/run; small differences are not evidence of statistical significance. Synthetic corruption scores do not establish robustness to real municipal vehicle footage, nor novelty by themselves. Repeated validation-guided selection creates selection bias; use a frozen protocol and untouched test set for final claims.

A fair next comparison would use matched training budgets and splits, report clean and corrupted validation performance together, and repeat promising pairs with multiple seeds. Do not tune on the internal test.

YOLO-RD was not run: exact implementation was unavailable. It is not counted as a completed model. No paper benchmark values are represented as our measurements.

## Excluded diagnostic runs

- `runs\gpu_smoke\results.csv`

- `experiments\detector_comparison_20261014\preflight\yolo11s\results.csv`

- `experiments\detector_comparison_20261014\preflight\yolo12s\results.csv`

- `experiments\detector_comparison_20261014\preflight\yolov10s\results.csv`

- `experiments\detector_comparison_20261014\preflight\yolov5su\results.csv`

- `experiments\detector_comparison_20261014\preflight\yolov6s_scratch\results.csv`

- `experiments\detector_comparison_20261014\preflight\yolov8s\results.csv`

- `experiments\paper_noise_20261010\gpu_preflight_20261010_184826\train\results.csv`

- `experiments\paper_noise_20261010\paper_augmentation_smoke_20261010_172804\train\results.csv`

- `experiments\paper_noise_20261010\paper_consistency_smoke_20261010_172821\train\results.csv`

- `experiments\paper_noise_20261010\paper_proposed_smoke_20261010_170415\train\results.csv`

## Machine-readable evidence

The accompanying JSON contains all included epoch rows, saved configuration, matched evaluation records including per-class corruption metrics, source paths and checkpoint hashes. CSV contains the compact main comparison.
