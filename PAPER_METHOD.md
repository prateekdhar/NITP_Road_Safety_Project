# Paper-guided robustness implementation

Date: 10 October 2026. Implemented against local Ultralytics 8.4.161. This is an experimental adaptation, not a reproduction of NER or PCSD and not an established novel method.

## Model choice

Student initialization: `runs/nrdd2024_yolov9s_noise_100_oct04/weights/best.pt` (SHA256 `18c5796b9f03c3440ec5f3fabd19bf2e63a1fd53e12612b1e087c03a96c3069b`). This is our strongest model with measured noise robustness: clean AP50 53.1268%, strong Gaussian noise AP50 50.1814%, compared with the unmodified YOLOv9s baseline's 38.9983% in that noise condition. This is not a claim that it wins every metric against every earlier model. The previous YOLO26s has not been included in this matched corruption comparison.

Frozen teacher: `runs/nrdd2024_yolov9s_baseline_100_oct04/weights/best.pt`, SHA256 `4e3731971fc26b01efca24cf69efd6a641a3c0a80d6b34a4d9303a8a0f0c1569`. It sees the geometrically aligned reference view before added corruptions. Its targets supplement the labels; it is not assumed to be more accurate on every class. Original checkpoints are read only and their hashes are checked before and after a run.

## What the complete PDFs establish

1. Guo et al., *Uncertainty-guided progressive learning for pavement crack segmentation under challenging visual degradation in smartphone-based imaging*, Measurement 273 (2026) 121178. DOI: https://doi.org/10.1016/j.measurement.2026.121178. Supplied `Measurement.pdf`.
   - Sections 3.1–3.3 and 4.2: teacher-versus-GT discrepancy defines difficulty; a pacing threshold admits harder pixels; the frozen teacher provides auxiliary KL guidance. Sections 4.1/4.2 use BCE/Dice on segmentation masks.
   - Section 5.3, PDF page 14: synthetic-to-real limitations and temporal continuity under vibration/blur remain concerns.
2. Duan et al., *NER: Noise-aware enhancement-assisted real-time detection transformer for pavement damage detection in hazy and low-light environment*, Expert Systems with Applications 323 (2026) 132469. DOI: https://doi.org/10.1016/j.eswa.2026.132469. Supplied `Expert systems.pdf`.
   - Sections 2.2–2.3: NDR-CPA restoration, noise-aware deformable branches, prompting, EMA-HG and AFPN within RT-DETR; this is substantially more than inserting a denoising filter.
   - Section 3.1: its public datasets are RDD2020 and the 8,000-image Baidu street-view SVRDD. **This SVRDD is not N-RDD2024.** Its self-built data exclude excessive blur. Its 78.0% AP50 is not our ten-class evaluation.
   - Section 5, PDF page 15 explicitly says truly unseen motion blur and other degradation types have not been evaluated. Deployment/compute constraints remain.

These findings support testing detection-relevant structure under new blur/noise. They do not establish that our method is novel or will beat either paper. Our lightweight training-only adaptation is chosen to reuse the proven student and avoid an unverified reconstruction of NDR-CPA or an entirely new RT-DETR training budget.

## Implemented variants

All variants use the same starting student, original ten-class split, seed, resolution and broader augmentation. Ordinary YOLO detection supervision remains active for **all** GT boxes, including those missed by the teacher.

- `augmentation`: broader corruption training only. This controls for extra epochs and augmentation.
- `consistency`: the same augmentation plus frozen-teacher sigmoid/Bernoulli KL and normalized multi-scale feature alignment on teacher assignments that match the GT class and have box IoU >= 0.3. Equal normalization per object prevents larger objects dominating.
- `proposed`: the same guidance, plus an easy-to-hard selection of reliable positive anchors (70% to 100% over 10 epochs), relative blur/object-size weighting capped at 3x, and two-axis local feature-difference agreement. These are weak feature-structure constraints, not segmentation or physical crack-width measurements.

The KL/feature/structure coefficients are 0.2/0.1/0.05 and ramp over ten epochs. They are initial engineering choices, not values reported optimal by either paper. No test metrics were used to choose them. GT detection loss is never weakened or masked. The teacher stays in eval mode with gradients disabled. All teacher operations and auxiliary losses disappear at inference; saved EMA checkpoints remain stock YOLOv9s.

Differences from PCSD: ten-class anchor/box detection replaces binary pixel segmentation; per-batch reliable-positive quantiles replace its full-training-set threshold; GT loss remains unchanged; there is no Dice loss because no pixel masks exist. Differences from NER: no NDRConv, CGM, LCPB, EMA-HG, AFPN or restoration network is claimed. Its damage-structure principle motivates the feature loss. Proposed scale weighting/local-structure loss are our experimental adaptation, with novelty still unverified.

## Corruptions and evaluation

Training retains 30% of images unchanged. Remaining samples receive horizontal/vertical motion blur (3/5/7-pixel kernels), heteroscedastic Gaussian RGB noise with dimming, or mixtures. All changes preserve box coordinates. A dedicated per-epoch generator matches corruption sampling between variants. Ordinary image/geometry augmentation is still present; mosaic/mixup/cutmix are disabled for this fine-tuning stage.

This first implementation corrupts tensors **after** resizing and padding. It is a practical synthetic approximation, not source-resolution sensor/ISP simulation; padded areas are also affected. It does not implement JPEG, rolling shutter, temporal fusion, measured camera noise or actual Poisson sensor sampling. These limits are deliberate and documented rather than claimed as completed features.

`evaluate_paper_study.py` compares chosen checkpoints on validation only, with clean, signal-dependent noise, held-out diagonal motion blur (length 9), disk defocus, and a diagonal blur/noise/low-light mixture. Inputs use the same random seed/batch ordering for each model. It saves precise aggregate/per-class metrics and predictions. Keep this suite fixed and do not repeatedly redesign it to favor a result. The original synthetic suite remains useful as a regression check, but its Gaussian noise scores are not the new sensor-noise scores.

All scientific comparisons must use the same training opportunity. An improvement over an unchanged checkpoint after extra epochs does not isolate the new method. At least the augmentation control is needed; the standard-consistency variant is needed to isolate the proposed weighting/structure contribution. Component ablations, three seeds and independent routes/cameras are still required for strong claims.

## Files and commands

Run from the RoadDamageDetection project using its `.venv` Python.

```powershell
.\.venv\Scripts\python.exe test_paper_robustness.py
.\.venv\Scripts\python.exe run_paper_study.py --smoke --device cpu --variant proposed
```

The smoke check copies eight training images into a separate dataset; its two validation images are also training images. Its scores are **not research results**. It checks forward/backward, validation, save/reload and unchanged source checkpoints. It uses a new timestamped output directory and never resumes or overwrites the previous runs.

Full fine-tuning requires an explicit epoch argument. Example only, not automatically launched:

```powershell
.\.venv\Scripts\python.exe run_paper_study.py --variant augmentation --epochs 30 --device 0
.\.venv\Scripts\python.exe run_paper_study.py --variant consistency --epochs 30 --device 0
.\.venv\Scripts\python.exe run_paper_study.py --variant proposed --epochs 30 --device 0
```

Each is 30 additional epochs, so all three total 90. Defaults: 640, batch 2, AdamW lr 0.0001, seed 42, fixed epoch cap and best clean-validation checkpoint selection. Training speed/VRAM feasibility at 640 with a teacher is unverified until a GPU smoke test. The script has no implicit restart or automatic queue; interrupted fine-tuning needs an explicitly audited resume implementation.

```powershell
.\.venv\Scripts\python.exe evaluate_paper_study.py --weights "reference=C:\path\reference\best.pt" "proposed=C:\path\proposed\best.pt" --device 0
```

All outputs go under `experiments/paper_noise_20261010`. Configuration and status files distinguish smoke tests from real runs. No held-out-test evaluation is initiated. No background training is launched by importing any module.
