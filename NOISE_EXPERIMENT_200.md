# Current experiment plan - revision 3

This supersedes the SR-WBCE plan in EXPERIMENTS_200.md. The user selected a matched comparison focused on image noise.

## Runs

1. YOLOv9s baseline: 100 epochs, already running and preserved.
2. YOLOv9s with additional noise augmentation: 100 epochs, queued automatically after the baseline.

Total budget: 200 epochs. Both start from the same weights/yolov9s.pt, with the same original ten-class N-RDD2024 split,640pixel input,batch4,seed42,optimizerauto and ordinary YOLO augmentations. Neither class reweighting nor rare-class oversampling is used. The old YOLO26s300epoch model remains preserved.

The baseline has standard YOLO augmentation; 'without noise robustness' means without our additional corruption transform, not no augmentation at all.

## Treatment

Training images, after ordinary preprocessing:50% unchanged;25% additive Gaussian sensor noise (standard deviation uniformly0.01-0.06 on0-1 images);25% low-light blur (brightness factor0.35-0.75,contrast0.8,3x3Gaussian kernel). Transformations preserve geometry and labels, clamp values to0-1, and retain clean examples. No inference denoiser or new architecture is added.

The motivation for low-light brightness/contrast reduction and mild Gaussian blur is Zhao et al., Robust road damage detection via multi-expert collaboration and cross-scale attention, Advanced Engineering Informatics72(2026)104452, Elsevier, section4.2. https://doi.org/10.1016/j.aei.2026.104452 . Local source: C:/Users/prate/Downloads/Robust.pdf.

Our implementation is an adaptation, not a reproduction of its multi-expert architecture, rain augmenter or evaluation splits. Numerical settings and the additional Gaussian sensor-noise stressor are experimental choices, not reported optimal values from that paper. Standard noise augmentation alone does not establish novelty.

YOLOv9s N-RDD2024 baseline evidence: YOLO-RD, Neurocomputing662(2026)132000, Table11, https://doi.org/10.1016/j.neucom.2025.132000 . Its seven-class evaluation differs from ours.

## Fair evaluation

Select each model's best checkpoint using original clean validation mAP50-95. After both finish, evaluate both on the same complete validation split, at640pixels, under clean, Gaussian noise0.03/0.06, and lowlight blur0.6/0.35. The corruptions are applied during evaluation without changing original files; the same seed and batch order produce identical degraded inputs for each model. Record mAP50,mAP50-95,precision and recall for every condition in experiments/oct04_200/robustness_comparison.json.

Compare paired changes and clean-performance cost. Synthetic validation measures this corruption family only; it cannot prove real-world robustness. No test-set tuning. Real degraded external footage and replicated seeds are needed for stronger claims. One seed cannot establish statistical significance.

## Files and queue compatibility

noise_training.py defines the trainer and evaluator. experiment_200.py dispatches it using revision3 of experiments/oct04_200/plan.json. The already-running parent retains the legacy second key 'yolo26s_balanced'; for revision3 that key explicitly dispatches YOLOv9s NoiseTrainer and saves to runs/nrdd2024_yolov9s_noise_100_oct04. Accordingly the second log/progress filenames retain the legacy key; trust the model/config/run name, not that filename. This avoids interrupting the baseline. plan_srwbce_not_run.json preserves the superseded plan.

Validation: numerical tests passed for deterministic noise, tensor shape preservation, finite outputs, valid pixel range and reduced brightness. The modified full training run has not yet begun; it starts only when the baseline completes. Monitor the second log for runtime failures.
