# Completed experiments and next research direction

Review date: 10 October 2026, Asia/Kolkata. This is a research recommendation, not a claim of a new state of the art. No additional training has been launched.

## What we have demonstrated

Two YOLOv9s runs completed 100 epochs each. The treatment adds synthetic noise/low-light blur during training; it does not change the architecture or apply a denoiser at inference. Both best checkpoints were selected on clean validation mAP50-95. Their CSV maxima occurred at epochs 97 and 98 respectively. The subsequent best-checkpoint evaluations are the source of the table below, rather than the final epoch or callback metrics.

| Validation condition | Baseline AP50 | Noise-trained AP50 | Change (percentage points) | Baseline AP50-95 | Noise-trained AP50-95 |
|---|---:|---:|---:|---:|---:|
| Clean | 52.57 | 53.13 | +0.56 | 30.46 | 30.91 |
| Gaussian noise, sigma 0.03 | 48.10 | 52.66 | +4.56 | 27.46 | 30.37 |
| Gaussian noise, sigma 0.06 | 39.00 | 50.18 | +11.18 | 21.99 | 28.97 |
| Low light, brightness 0.60, fixed blur | 49.38 | 53.23 | +3.85 | 28.22 | 30.72 |
| Low light, brightness 0.35, fixed blur | 44.61 | 52.75 | +8.14 | 25.25 | 30.38 |

Across the four synthetic conditions, the equal-weight mean AP50 rises from 45.27% to 52.21% (+6.93 points); mean AP50-95 rises from 25.73% to 30.11% (+4.38 points). This is a descriptive summary of our four conditions, not a standardized benchmark score. Under strong Gaussian noise, AP50 retains about 94.5% of clean performance, compared with about 74.2% for the baseline.

Full precision metrics, precision/recall for every condition, checkpoint hashes, histories, per-class console results, settings and provenance are in `statistics.json` and `statistics.txt`. The original 300-epoch YOLO26s report is also archived under `records/previous_300epoch_*`; its held-out-test results must not be mixed with these validation results.

### What the current experiment cannot establish

- The validation corruptions belong to the same families and parameter ranges used in training. This is evidence of in-family robustness, not unseen-noise generalization.
- The two low-light tests use the same 3x3 Gaussian blur; only brightness changes. Neither is a directional motion-blur evaluation. Noise and low-light blur were sampled separately in training; combined noise + motion + darkness remains untested.
- Corruptions were added after resizing/letterboxing, including padded pixels. That differs from physical camera exposure, sensor noise, image processing and video compression.
- One seed was run; the noise run resumed after epoch 56. Resumption can change the random-number trajectory. No repeated-seed confidence interval or significance claim is justified.
- Thin-crack preservation has not been measured directly. N-RDD2024 boxes are not crack masks and do not supply physical crack width.
- The validation set has only three D90 boxes, and the internal test has one. D90 AP and precision are unstable; console precision of 1 with recall 0 is not useful detection performance.
- Training progress flattens late: epoch-80 AP50 was 52.16% baseline and 53.09% noise-trained. More epochs alone are not the clearest next intervention. Both clean AP50-95 maxima occur near the end, so this is not proof of severe overfitting.
- Rounded clean per-class AP50 suggests improvements for D80 and D50, but D30 decreases by about 1.7 points. Aggregate improvements do not mean every damage category improves. Prediction-level error inspection is needed before assigning causes.

## The original three papers: evidence and remaining questions

### 1. Wang et al., YOLO-RD — Neurocomputing 662 (2026), 132000, Elsevier

[Publisher / DOI](https://doi.org/10.1016/j.neucom.2025.132000). Reviewed publisher metadata and the supplied `YOLO.pdf`, especially section 4.6.3/Table 11 (PDF page 17) and conclusion (page 19).

The paper combines fine-feature extraction, contextual attention and class-balanced loss. Its conclusion explicitly identifies multi-source noise/background interference and extreme few-shot adaptation as limitations. This supports our noise research motivation. It does not imply noise robustness was absent from all previous research.

Table 11 reports YOLOv9s 46.9/26.9 and YOLO-RD 45.5/26.2 AP50/AP50-95. It retains seven classes and uses 12,625/1,750/3,492 train/validation/test images after cleaning, using the original validation pool for testing. Our ten-class, 13,782/3,704/1,509 protocol differs. Our 53.13% does **not** demonstrate superiority over their 45.5%.

**Candidate extension (our inference):** characterize failure versus object size and mixed noise/blur, then train a mechanism that preserves detection-relevant structure. Reproduce their protocol or retrain their released/reproduced method on ours before making a direct comparison. The paper already includes vehicle deployment and location reporting; those features alone are not a new research contribution.

### 2. Zhao et al., Robust road damage detection via multi-expert collaboration and cross-scale attention — Advanced Engineering Informatics 72 (2026), 104452, Elsevier

[Publisher / DOI](https://doi.org/10.1016/j.aei.2026.104452). Reviewed supplied `Robust.pdf`, section 5.2 (PDF pages 12–13).

The authors already address adverse conditions with expert collaboration and cross-scale attention. They explicitly acknowledge that simulated low-light/rain scenes do not fully represent extreme real environments, and that added modules increase computation. They propose real dashcam collection and teacher–student compression as future work. Their CNRDD/RDD2022/ARSDD experiments are not our N-RDD2024 protocol.

**Candidate extension (our inference):** real camera noise/blur evaluation and a student that needs no heavy restoration branch at inference. Distillation is explicitly suggested by this paper, so applying ordinary distillation alone is an extension, not persuasive novelty. The technical contribution must be more specific and beat a standard-distillation control.

### 3. Hao and Li, Development and multi-scenario validation of a real-time pavement distress monitoring system based on lightweight YOLOv8 — Discover Applied Sciences 8, 955 (2026), Springer Nature

[Full article](https://link.springer.com/article/10.1007/s42452-026-08970-x). Published 12 June 2026; reviewed full publisher text.

Its improved YOLOv8 uses a dynamic feature pyramid, attention and pruning. Sections 5.4 and the failure discussion describe problems with small cracks, reflective/snow scenes, rapid-motion blur and worn-road textures. It also already uses brightness/filtering preprocessing and blur augmentation. Its reported 78.5% mAP comes from a different RDD2020/self-collected protocol.

**Candidate extension (our inference):** evaluate damage-preserving suppression of camera noise and directional blur, including false positives on intact rough asphalt. Treat it as supporting application literature, not the main numerical target. Reproducibility deserves scrutiny: the text contains inconsistent-looking class counts and hardware descriptions, and median filtering should not be assumed to reverse motion blur merely because it is described that way. A publisher's reputation does not replace methodological checks.

## Recent work that changes the novelty assessment

The following are primary publisher/proceedings or author sources. “Scope gap” means the reviewed evidence does not establish the desired result; it does not prove the entire paper omitted it.

| Source and status | Relevant evidence | Implication for our design |
|---|---|---|
| **Duan et al., NER: Noise-aware enhancement-assisted real-time detection transformer for pavement damage detection in hazy and low-light environment. Expert Systems with Applications 323 (2026), 132469, Elsevier.** [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417426013825) / [DOI](https://doi.org/10.1016/j.eswa.2026.132469) | Publisher-indexed text describes adaptive enhancement, noise suppression and detection-oriented processing with an enhanced RT-DETR. Its stated scope is the tested hazy/low-light settings, with broader real conditions needing validation. Full paper was not obtained in this review. | “Noise-aware enhancement for road damage” is already covered. This is a priority comparator to inspect in full before freezing our method; its reported numbers are not assumed to use our data or split. |
| **Guo et al., Uncertainty-guided progressive learning for pavement crack segmentation under challenging visual degradation in smartphone-based imaging. Measurement 273 (2026), 121178, Elsevier.** [Publisher](https://www.sciencedirect.com/science/article/pii/S0263224126008870) / [DOI](https://doi.org/10.1016/j.measurement.2026.121178) | Publisher abstract describes uncertainty-paced curriculum self-distillation and supervision from labels and teacher outputs, with illumination/shadow/blur experiments. Abstract/preview reviewed, not full method. | Curriculum + uncertainty + distillation already exists for cracks. Our box-detection setting differs, but a task change by itself is weak novelty. Include an ordinary consistency/distillation baseline. |
| **Wang et al., Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection (JFD3). AAAI 40(12), 10181–10189 (2026).** [Paper page](https://ojs.aaai.org/index.php/AAAI/article/view/37986) / [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37986/41948) | Peer-reviewed conference, not a journal. Clear/blurred branches, feature restoration, frequency guidance and consistency learning; infrared UAV domain. | Strong support for task-level feature recovery, but it prevents claiming clean-to-blurred feature consistency is new. RGB road textures and small damage still need their own evidence. |
| **Guo et al., DFFormer: Dual Frequency-Driven Transformer for real-world image deblurring. Computer Vision and Image Understanding 268 (2026), 104763, Elsevier.** [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S107731422600130X) / [DOI](https://doi.org/10.1016/j.cviu.2026.104763) | Publisher abstract describes spatial/frequency processing, FFT/DWT and restoration on real blur in photography and pathology. Abstract/preview reviewed. | Useful restoration reference; image quality gains do not establish road-detection AP gains. Frequency processing can retain both cracks and noise: test the detection outcome. |
| **Xiao et al., Self-Supervised Visual Prompting for Cross-Domain Road Damage Detection (PROBE). IEEE/CVF WACV (2026), 3514–3524.** [Proceedings](https://openaccess.thecvf.com/content/WACV2026/html/Xiao_Self-Supervised_Visual_Prompting_for_Cross-Domain_Road_Damage_Detection_WACV_2026_paper.html) / [PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Xiao_Self-Supervised_Visual_Prompting_for_Cross-Domain_Road_Damage_Detection_WACV_2026_paper.pdf) | Peer-reviewed conference, not a journal. Defect-aware prompting and domain alignment use unlabeled target data with a frozen ViT. | Cross-camera/region adaptation is also studied. If we use target-camera footage for adaptation, disclose it and keep those routes out of evaluation; do not compare that setting with source-only training as if identical. |
| **Vo and Park, Deep Joint Unrolling for Deblurring and Low-Light Image Enhancement (JUDE). IEEE/CVF WACV (2025), 2696–2705.** [Proceedings](https://openaccess.thecvf.com/content/WACV2025/html/Vo_Deep_Joint_Unrolling_for_Deblurring_and_Low-Light_Image_Enhancement_JUDE_WACV_2025_paper.html) | Accepted conference paper with accessible PDF. Joint low-light/blur modeling and noise removal evaluated on LOL-Blur/Real-LOL-Blur. | A useful older, inspectable physical-model reference. Restoration scores are not directly comparable with our AP. |

### Closely overlapping preprint: do not ignore it

[RMR-P: Road Metadata-Aware Restoration for Pavement Inspection](https://arxiv.org/abs/2608.08957v2), Ghorbani et al., August 2026, is a preprint submitted to ICCAIS, not a verified journal publication. Search results still show the earlier title “RMR-Net.” The current v2 record describes degradation-conditioned road restoration, detail preservation and evaluation through a frozen YOLO11s detector. Experiments use IVCNZ/PCM and known synthetic degradation parameters. It materially overlaps a generic proposal to preserve road edges with noise-aware restoration. Cite it as prior work when assessing novelty, even though it does not satisfy the preferred journal evidence requirement.

### IEEE journal lead requiring full-text verification

Xiao et al., *A Model-Driven Deep Neural Network for Simultaneous Low-Light Image Enhancement and Deblurring*, DOI [10.1109/TNNLS.2026.3724469](https://doi.org/10.1109/TNNLS.2026.3724469), is indexed as IEEE TNNLS early access, 26 August 2026. The publisher page was inaccessible in this review; the [author repository](https://github.com/cassiopeia-yxx/MDNet) is accessible but still says weights will follow acceptance. Keep this as a retrieval lead, not a verified reproducible baseline or support for numerical superiority. No pretrained-weight availability or Windows compatibility is assumed.

## Recommended candidate: damage-scale-conditioned consistency learning

**Research question:** Can a small RGB detector retain object-localization and crack evidence under unfamiliar combinations of camera noise and motion blur, without adding a restoration network at deployment?

**Potential contribution, not yet demonstrated:** a training objective that conditions *where and how strongly* to enforce clean/degraded agreement on the relative severity of blur to the damage's projected scale, while avoiding transferring unreliable clean-teacher predictions. Its distinctiveness must be checked against NER, PCSD, JFD3 and related papers in full. No claim of being first is currently justified.

Proposed implementation:

1. **Create physically motivated paired views.** Apply the same geometric transform to the clean reference and degraded student view. Simulate directional motion trajectories, defocus, signal-dependent shot/read noise, low light and JPEG/video-like compression at source resolution before letterboxing. Include mixtures and retain clean examples. On processed RGB this is still an approximation to a sensor pipeline; calibrate it using actual camera samples before claiming physical realism.
2. **Use a clean teacher during training only.** A frozen clean YOLOv9s supplies detached feature targets on the paired clean view. It is not an oracle: gate soft targets by agreement with ground-truth boxes/classes and by teacher stability. Always keep ground-truth detection loss, including for objects the teacher misses. Teacher forward passes use no gradients; batch 1–2 and AMP are plausible starting points on the 8GB RTX4060, subject to measured memory.
3. **Test object-relative degradation weighting.** Map boxes to shallow/mid-level features. Weight alignment by projected box scale and the sampled blur extent rather than using one global consistency weight. For elongated D00/D10 boxes, compare structure along and across the dominant local orientation; treat this as a weak structural proxy, not measured crack topology. Discard invalid orientation cues on potholes/alligator cracks. Gate teacher targets when the clean teacher is unreliable, and avoid forcing impossible reconstruction of fully erased structure.
4. **Learn localized agreement.** Combine ordinary detection loss, normalized feature agreement within reliable object regions, and a modest directional-structure penalty. Ground-truth supervision remains active everywhere. Use safe normalization so large boxes do not dominate. Avoid raw high-frequency amplification, which can strengthen sensor noise and asphalt texture as well as cracks.
5. **Remove all teaching components at inference.** Start with an unchanged YOLOv9s student. This makes added deployment cost essentially zero apart from any later verified changes. If training-only consistency fails, test one identity-initialized shallow feature adapter as a separate experiment, measuring its extra latency and clean-image cost. Do not begin by combining multiple attention/restoration modules.

This is an experimental design proposed here, not a method claimed by the cited papers. Its strongest possible differentiator is the object-relative, directional degradation weighting plus evidence on unseen mixed corruptions and independent routes. Reusing YOLOv9s, distillation, corruption augmentation, edge losses or N-RDD2024 alone is not novel.

### Why this is preferable to simply adding a denoiser

Noise and thin cracks both occupy high-frequency image content. A denoiser trained for visually smooth output can erase a true crack; a sharpener can amplify false crack-like texture. Training against detection and reliable object features targets the task directly. The student-only inference design also fits vehicle deployment better than a large restoration model. These are hypotheses motivated by prior work, not guaranteed AP gains.

## Ranked experiments and fair comparisons

| Priority | Experiment | What it isolates |
|---|---|---|
| 1 | Evaluate the existing two checkpoints on a frozen, broader validation corruption suite: directional motion, defocus, shot/read-like noise, compression and mixtures | Whether the current gain generalizes at all; no new training needed |
| 2 | Broader physically motivated augmentation alone | A strong, simple control; it may explain most gains |
| 3 | Same augmentation + ordinary clean/degraded consistency | Benefits already expected from prior feature-learning work |
| 4 | Same setup + proposed object-relative reliability/directional weighting | The candidate technical contribution; compare directly with experiment 3 |
| 5 | Ablate reliability, size conditioning and direction separately; repeat promising comparisons over at least three seeds | Whether each proposed component matters and gains are stable |
| 6 | Reproduce accessible YOLO-RD/NER or a documented component adaptation, and evaluate a generic restoration baseline | Whether improvement survives comparison with literature-based alternatives |

Use equal epochs, initialization, resolution, image exposures and checkpoint-selection rules for paired training controls; report GPU-hours as well because teacher training costs more. If fine-tuning existing checkpoints, give every comparator the same additional training opportunity and label the study as fine-tuning. Keep higher resolution, rare-class reweighting and architecture changes separate initially so they cannot masquerade as noise-method gains. Higher-resolution inference is a useful AP/latency ablation, but it is not the proposed novelty.

For final model selection, predeclare a validation robustness objective and a clean-performance constraint. An example engineering gate is at least +2 AP50 points on held-out corruption families versus the broader-augmentation control, no more than 0.5 point clean AP50 loss, non-decreasing AP50-95, and acceptable batch-1 latency. These are proposed acceptance criteria, not predictions, established standards or evidence of statistical significance. Freeze the criterion before launching comparisons; report results even when it fails.

## Evaluation needed for a defensible paper

- Preserve all ten classes in the primary current-protocol table. Report per-class AP/support, AP50 and AP50-95, precision/recall at a predeclared confidence rule, false positives on intact rough asphalt, and object-size bins. Do not suppress D90 just to raise mAP; any seven-class result must be a separate declared experiment.
- Reserve corruption families/trajectories/severities for evaluation. Keep exactly matched degraded inputs for all models using saved manifests and hashes. Training-family robustness, unseen-family robustness and real-camera results need separate tables.
- Split real footage by route/session/camera, not adjacent random frames. Include genuinely noisy/blurred footage and clean controls, with human-reviewed boxes. Exclude unobservable defects from confident labeling or mark uncertainty under a written annotation policy.
- Paired bootstrap confidence intervals must recompute AP from predictions and labels, preferably clustered by route/session. Epoch variation and the spread between corruption conditions are not statistical confidence intervals. Use multiple training seeds too.
- Record batch-1 end-to-end preprocessing/inference/postprocessing latency, p50/p95, memory and sustained throughput on the RTX4060. Camera capture, GPS and upload costs are separate deployment measurements. Our current batched validation timing does not prove live FPS.
- The current validation set has already informed development. Freeze the final method before evaluating the internal held-out test; never tune against it. Because the earlier YOLO26s test result has already been inspected, document that history and seek an independently collected final route/camera test for stronger claims.
- To claim superiority over a paper, match dataset release, class mapping, cleaning, split, image size and AP definition, and reproduce comparable baselines. Either use the authors' exact split when available, or retrain their method under our common protocol and describe it accurately. Comparing 53.13% on our validation with 45.5%, 78.5% or 90% on unrelated evaluations is invalid.

## Practical decision

Continue this project. The completed runs provide a useful baseline and a measurable noise benefit. First establish unseen motion-blur/combined-degradation performance, then compare strong augmentation, standard consistency and the proposed weighting under matched conditions. Collect independent real-camera evidence in parallel. More epochs or an arbitrary larger model would not answer the research question.

There is no credible basis yet to promise 70%, 80% or any particular final AP. There is also no evidence yet that the proposed mechanism is novel or better than NER/YOLO-RD. The next study should be designed to test those claims, not assume them.

Access notes: original Elsevier papers were checked from user-supplied PDFs; Springer full text and AAAI/CVF records were accessible. NER, PCSD and DFFormer findings above are limited to publisher-indexed abstracts/previews. No paywall was bypassed, no new model code was installed and no new training budget was committed.
