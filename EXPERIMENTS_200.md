# Journal-backed 200-epoch experiment plan

User budget: 200 epochs TOTAL, split into 100 epochs per model, sequentially on RTX4060. Keep the previous 300-epoch run intact. No early stopping is configured in this fixed-budget comparison; each run saves its best validation checkpoint.

## Evidence

- Wang et al., **YOLO-RD: Road defect detection with context-aware attention and balanced loss**, Neurocomputing662(2026)132000, Elsevier. https://doi.org/10.1016/j.neucom.2025.132000 . Local source: C:/Users/prate/Downloads/YOLO.pdf. Table11 reports YOLOv9s46.9% mAP50 and26.9% mAP50-95 on its seven-category N-RDD2024 split. This motivates a YOLOv9s baseline; our ten-category split differs.
- The same paper, Equation17, defines SR-WBCE as per-class BCE weighted by (max class count / class count)^(1/n). Section4.4.3 reports its preferred n=4. We use training bounding-box counts only and apply the weight to the entire BCE per category, including negative terms, without clipping/normalization. This is an implementation of the published loss rule adapted to BOTH YOLO26 training heads, not a reproduction of the complete YOLO-RD architecture or a guarantee of improvement.
- Tornero-Soria et al., **Toward a Deeper Understanding of YOLO26: Block-Level Architectural Analysis and Ablation Studies**, Applied Sciences16(13)6758(2026), MDPI. https://doi.org/10.3390/app16136758 . This is a journal architecture analysis, primarily YOLO26n/COCO; it does NOT prove YOLO26s superiority on N-RDD2024. MDPI is a different publisher from the preferred Elsevier/Springer/IEEE sources. The modification itself comes from the Elsevier source above.

## Executable plan

1. YOLOv9s pretrained baseline,640 input,100epochs.
2. YOLO26s pretrained detector with fourth-root SR-WBCE classification weighting,640 input,100epochs.

Both use the original training/validation split, batch4, seed42, AMP, automatic optimizer selection, and identical basic augmentations. The original detector is preserved. Automatic optimizer selection can vary by architecture; capture actual optimizer in logs. The draft832pixel/oversampling/AdamW recipe was superseded before training; plan_provisional_not_run.json and unused balanced manifest remain only as provenance. Neither is used in the final revision2 plan.

This two-model experiment compares practical candidates, but cannot isolate the causal contribution of the loss because architectures differ. A matched YOLO26s100epoch unweighted control would require another authorized experiment. Comparison to the existing300epoch run is descriptive, not a controlled loss ablation.

Validation only is used for selection. No test evaluation or parameter tuning on the previous test set occurs in this queue. The rarest class has40 training boxes and3 validation boxes, so its validation result is unreliable. Future deployment claims require an independent route-separated test.

## Outputs and checks

Runner: experiment_200.py. Plan and sequential logs: experiments/oct04_200/. Models: runs/nrdd2024_yolov9s_baseline_100_oct04 and runs/nrdd2024_yolo26s_srwbce_100_oct04. Completion summaries include best checkpoint hashes and per-class validation AP/precision/recall. Existing runs are not overwritten. A queue lock prevents concurrent queue launches; a failed model halts the queue.

Numerical preflight compares weighted BCE with an explicit per-class implementation, checks finite gradients, and runs CUDA forward/backward for both ten-class architectures. This consumes no training epochs and changes no saved model weights.
