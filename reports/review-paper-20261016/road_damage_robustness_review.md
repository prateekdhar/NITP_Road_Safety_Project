# Road Damage Detection under Visual Degradation: A Critical Review of Robust Learning, Image Enhancement, and Deployment Evidence

Prateek Dhar and Ashish Kumar

Department of Computer Science and Engineering, National Institute of Technology Patna

Guide: Dr. Amrita Mohan

## Abstract

Camera-based road inspection offers a practical route to scalable pavement monitoring, but image noise, blur, low illumination, and atmospheric interference can undermine defect recognition. Recent research addresses these limitations through context-aware detection, specialized expert networks, uncertainty-guided learning, and enhancement-assisted transformers. This focused narrative review critically examines four 2026 Elsevier research articles alongside two Springer reviews, with supporting dataset and architectural literature. The synthesis distinguishes bounding-box detection from pixel-level crack segmentation and evaluates the relationship between algorithmic improvements, degradation coverage, benchmark comparability, and deployment evidence. Reported accuracy gains are informative within individual experimental protocols, but differences in class definitions, dataset partitions, and evaluation tasks prevent a defensible cross-paper ranking. In particular, performance on a seven-class N-RDD2024 subset cannot be treated as equivalent to performance on its ten-class release. The reviewed literature already establishes noise-aware enhancement and robustness-oriented learning; consequently, adding augmentation or attention alone is an insufficient novelty claim. More defensible research directions include generalization to unseen and compound degradations, preservation of small damage structures, reliability across video sequences, and validation of complete inspection pipelines under resource constraints. A controlled evaluation framework is proposed that separates clean accuracy, corruption sensitivity, real-world transfer, and operational utility. The review concludes that progress should be demonstrated through reproducible, task-aligned robustness evidence rather than isolated improvements in aggregate detection scores.

Keywords: road damage detection; pavement inspection; visual degradation; noise robustness; motion blur; N-RDD2024; object detection; knowledge distillation

## 1. Introduction

Automated pavement inspection connects visual recognition with a practical maintenance decision: where a defect is located, what type of defect it represents, and whether it warrants inspection or repair. A camera mounted on a routinely operated vehicle can provide repeated observations without requiring a dedicated survey for every road segment. However, a recognizable defect in a carefully selected still image may become difficult to identify during vehicle motion, poor lighting, or adverse weather. The relevant research problem therefore extends beyond maximizing accuracy on a fixed image collection.

Recent surveys place deep learning within a broader road-inspection landscape. Abdelwahed et al. [1] review real-time detection methods and datasets, including detector families and computational considerations. Hemdan and Al-Atroush [2] discuss intelligent crack detection together with Internet of Things and digital-twin integration. These perspectives motivate attention to both visual recognition and its place in an operational inspection system. Neither a detection score nor the use of a recent model, by itself, establishes that a system can reliably support municipal maintenance.

Four recent journal studies provide complementary approaches to this problem. YOLO-RD improves context modeling, feature processing, and class-balanced learning [3]. A multi-expert framework introduces specialized feature extraction and cross-scale attention [4]. Paced Curriculum Self-Distillation (PCSD) addresses difficult crack pixels under degraded smartphone imagery [5]. Noise-aware enhancement-assisted real-time detection transformer (NER) couples image restoration with pavement damage detection under haze and low illumination [6]. These contributions make a broad claim that “noise has not been addressed” untenable. The more useful question is which kinds of degradation, transfer conditions, and operational outcomes remain insufficiently demonstrated.

This review contributes a task-aware synthesis of these approaches, an audit of the comparability of their evidence, and a research agenda for evaluating robust vehicle-based inspection. Its emphasis is on the boundary between what a reported experiment establishes and what remains a plausible but untested extension. The proposed evaluation framework is a methodological recommendation; no new model performance is reported.

## 2. Review scope and evidence assessment

This article is a focused critical narrative review, not an exhaustive systematic review or meta-analysis. The core corpus comprises four 2026 Elsevier research articles [3–6] selected for their direct relevance to road damage recognition, degraded imaging, or efficient deployment, and two Springer reviews [1,2] that establish the wider methodological and systems context. Supporting sources address dataset provenance and an earlier segmentation architecture [7–9]. Bibliographic details and publisher records were checked on 16 October 2026. Publication years follow the journal issue where available: reference [2] appeared online in 2025 and is assigned to a 2026 volume.

Full texts of the four core research articles and the open-access review [1] informed the synthesis. For the subscription review [2], the accessible abstract and bibliographic record support only its broad thematic positioning; its full-text methods and conclusions were not independently assessed. References appearing within a review were not automatically treated as primary evidence. The selection is purposive and does not support claims about the prevalence of a technique across the entire field. No database-wide search count, screening flow, or quantitative pooled effect is claimed.

For each research study, the assessment considers five questions: What is the prediction task? Which datasets and class definitions are used? Which adverse conditions are actually evaluated? What does the comparison control? What evidence supports deployment? Author-acknowledged limitations are distinguished from the methodological implications drawn in this review. Publication in an established journal is a useful provenance criterion, but does not remove the need to inspect evaluation design, reproducibility, and the scope of the reported results.

## 3. Tasks, degradations, and benchmark comparability

### 3.1. Detection and segmentation answer different questions

Bounding-box detection identifies damage instances through category labels and spatial boxes. Pixel-level segmentation identifies the image regions assigned to a defect. Their outputs overlap in purpose but are not interchangeable. A detector can localize a crack without recovering its width or continuity; a segmenter can preserve crack geometry without distinguishing the multiple defect categories required by a maintenance inventory. DeepCrack provides an earlier example of hierarchical, multi-scale feature learning for crack segmentation [8], while PCSD extends the discussion toward learning under challenging image conditions [5].

Detection precision measures the proportion of predicted detections that are correct under a specified matching rule; recall measures the proportion of annotated objects recovered. Average precision summarizes a precision–recall relationship. mAP50 averages class AP at an intersection-over-union threshold of 0.50, whereas mAP50–95 additionally averages over thresholds from 0.50 to 0.95. The latter places greater emphasis on localization quality. Segmentation IoU and Dice evaluate pixel overlap. An IoU or F1 value from a segmentation study cannot be substituted for detection mAP. All reported comparisons therefore require an explicit task and metric definition.

### 3.2. Visual degradation is not a single variable

Noise introduces unwanted intensity variation; motion and defocus blur suppress or redistribute spatial detail. Low illumination can reduce contrast and increase acquisition noise, while haze changes contrast through atmospheric effects. Shadows and wet reflections can resemble or obscure damage without behaving like additive noise. These distinctions matter because a method that suppresses random noise may also remove fine crack structures, and a model trained on synthetic haze need not tolerate vehicle-induced motion blur.

The taxonomy used here separates acquisition effects, environmental interference, and scene ambiguity. It also distinguishes individual degradations from compound conditions, such as low light combined with motion blur and compression. This is an analytical framework for organizing experiments, not a claim that the reviewed studies tested every category. A robustness claim should specify the corruption family, severity, whether it was observed during training, and whether the evaluation images were synthetically altered or captured under real conditions.

### 3.3. Dataset identity must be preserved

N-RDD2024 version 3 is an edited and expanded road-defect resource containing ten labeled categories: longitudinal, transverse, alligator, and repaired cracks; potholes; pedestrian-crossing and lane-line blurs; manhole covers; patchy road sections; and rutting [7]. The marking-related “blur” labels describe defect categories and should not be interpreted as annotations of camera motion blur. RDD2022 is a separate multinational dataset documented in a dedicated dataset paper [9]. Dataset names alone are insufficient to establish equivalent experimental conditions.

Wang et al. [3] evaluate a cleaned N-RDD2024 subset containing seven categories, excluding the two marking categories and manhole covers. Their reported partition contains 12,625 training, 1,750 validation, and 3,492 test images. The article describes repurposing the original validation set as a test set after cleaning and drawing a new validation subset from training data. These details, rather than an approximate split ratio, define the comparison. A ten-class experiment using another partition cannot claim superiority merely because its mAP is numerically higher.

Likewise, the SVRDD resource used in NER [6] must not be conflated with N-RDD2024. A benchmark comparison should preserve the dataset release, class map, image exclusions, split identifiers, and annotation policy. For video-derived data, route or sequence separation is especially important: adjacent frames can contain nearly identical damage observations, making randomly separated images a weak test of transfer to new roads.

## 4. Recent approaches to robust pavement recognition

### 4.1. Context-aware and balanced detection: YOLO-RD

Wang et al. [3] develop YOLO-RD around a YOLOv8s baseline, combining Convolution Spatial-to-Depth Attention Fusion, Local–Global Enhanced Context Attention, a smoothed-root weighted binary cross-entropy loss, and lightweight feature rearrangement. The design addresses irregular defect appearance, feature fusion, and category imbalance. Its relevance to degraded inspection lies in strengthening damage representation while controlling model complexity; it should not be reclassified as a dedicated motion-deblurring method.

Within the paper’s RM-RDD experiment, YOLO-RD reports 66.7% mAP50 and 41.2% mAP50–95, compared with 64.8% and 39.5% for YOLOv8s. However, the ranking changes on the seven-class N-RDD2024 experiment: YOLOv9s reports 46.9% and 26.9%, while YOLO-RD reports 45.5% and 26.2%. Table 2 preserves this distinction. The proposed architecture improves on its stated baseline without being the highest-scoring model on every dataset. This is evidence for dataset-dependent trade-offs, not a contradiction to be concealed.

The article also reports a vehicle deployment operating at 12 frames per second on a NOVATEK NT98525 platform after pruning and quantization [3]. This is stronger deployment evidence than parameter counts alone, but the device-specific result should not be generalized to an unmodified implementation or different hardware. Its future-work discussion identifies remaining difficulty with noise, complex backgrounds, and limited examples. An extension should therefore test a clearly specified failure mode while retaining the efficiency rationale of the original design.

### 4.2. Specialized experts and cross-scale attention

Zhao et al. [4] introduce a shallow detail-perceptive mixture-of-experts module and mixture-of-experts gated cross-scale attention. The rationale is that diverse defect structures and contextual relationships may benefit from specialized processing rather than a uniform representation. Expert-oriented augmentation and load balancing support the learning process. The reported mAP50 improvements over the baseline are 4.5, 6.6, and 4.2 percentage points on CNRDD, RDD2022, and ARSDD, respectively; these are within-study improvements, not transferable estimates for N-RDD2024.

The paper’s limitations are particularly relevant to robustness claims. Low-light and rainy scenarios are simulated through augmentation, and the authors acknowledge that synthetic data cannot fully reproduce real extreme environments. They also identify additional computational overhead and propose real dash-camera data collection and lightweight deployment as future work [4]. Thus, specialized experts offer a credible mechanism for heterogeneous features, but evaluation on augmented imagery does not establish unrestricted generalization to natural adverse conditions.

An additional question raised by this review is whether expert routing remains useful when several degradations occur together or when an unfamiliar camera changes image statistics. Attention visualizations can illustrate model behavior, but they cannot replace held-out quantitative tests of these conditions. Ablations should separate the contribution of expert capacity, augmentation, and routing, since all three can affect the final score.

### 4.3. Progressive learning and self-distillation

Guo et al. [5] present PCSD for pavement crack segmentation in smartphone-based imaging. A teacher model supplies a prior for ranking pixel difficulty, and training progresses from easier to harder pixels while using ground-truth and teacher-derived supervision. The difficulty quantity is based on discrepancy between teacher prediction and the ground-truth label. Consequently, it is a supervised training signal, not an uncertainty estimator that can be calculated identically on unlabeled deployment images.

The study is useful for understanding how difficult, low-contrast structures can be introduced without destabilizing optimization. Its limitations identify the synthetic-to-real gap, temporal continuity across video frames, and blur associated with vehicle vibration [5]. These observations support investigation of temporally reliable learning. They do not imply that transferring the same loss directly to a detector will reproduce the segmentation result: boxes, object assignments, background regions, and pixel masks impose different supervision requirements.

For detection, a teacher–student extension would need to define how object correspondence and confidence filtering are handled. It would also need to prevent incorrect teacher predictions from overriding trustworthy annotations. These are proposed adaptation requirements. The existing PCSD method remains prior art for curriculum learning and self-distillation under degraded pavement imagery; combining these generic ideas is not, by itself, an original contribution.

### 4.4. Enhancement-assisted transformer detection

Duan et al. [6] propose NER, combining noise-aware deformable restoration with content prompt aggregation and an enhanced RT-DETR-based detector. The detection component incorporates an efficient multi-scale attention block and an asymptotic feature pyramid. The approach explicitly treats image enhancement as relevant to the downstream recognition task, rather than assuming that a visually improved image will necessarily improve detection.

On the self-built evaluation dataset, the paper reports precision of 80.9%, recall of 79.2%, F1 of 80.0%, and mAP50 of 78.0% [6]. These values characterize that experiment; they cannot be ranked directly against the N-RDD2024 values in Table 2. The paper also evaluates RDD2020 and SVRDD. Its conclusions explicitly limit the evidence for truly unseen degradation types, including motion blur, rain streaks, lens flare, wet reflections, and snow, and identify embedded deployment as unresolved.

NER therefore provides especially direct prior art against a generic novelty claim based on noise-aware preprocessing plus detection. A stronger extension would establish when enhancement preserves defect evidence, when it introduces harmful artifacts, and how a system behaves outside the degradations represented in training. A clean-image bypass or a lightweight alternative is a candidate for controlled testing, not an established improvement.

## 5. Comparative synthesis and strength of evidence

Table 1 summarizes the complementary roles of the six core sources. The four research articles attack different parts of the recognition pipeline. Context-aware learning addresses representation; expert specialization addresses heterogeneous features; progressive learning addresses optimization difficulty; and enhancement-assisted detection addresses degraded input. Their mechanisms may be combined, but adding modules increases the need for ablation and latency accounting. A larger composite model is not automatically more robust or more deployable.

Table 1. Scope and evidence boundaries of the core literature

| Study | Approach or scope | Evidence boundary |
| --- | --- | --- |
| Abdelwahed et al. [1] | Review of real-time detection methods and datasets | Broad context; not a new controlled degradation experiment. |
| Hemdan and Al-Atroush [2] | Review of crack detection and intelligent monitoring systems | Abstract-level context in this review; full text not assessed. |
| Wang et al. [3] | Context-aware YOLO-RD and balanced loss | Multiple datasets and edge demonstration; accuracy ranking depends on dataset. |
| Zhao et al. [4] | Specialized experts and cross-scale attention | Synthetic adverse-condition evaluation; natural extreme conditions and compute remain concerns. |
| Guo et al. [5] | Pixel-level curriculum and self-distillation | Segmentation evidence; transfer to box detection and video continuity require evaluation. |
| Duan et al. [6] | Noise-aware restoration and transformer detection | Haze and low-light evidence; unseen motion blur and embedded operation not established. |

Synthesis of [1–6]. Boundaries distinguish reported limitations from the scope of evidence available to this review.

Table 2 reproduces selected numerical results from one common experimental setting in Wang et al. [3], rather than assembling a misleading leaderboard across unrelated datasets. YOLOv9s provides the highest mAP in this comparison, while YOLO-RD uses fewer reported parameters and GFLOPs. The distinction supports considering both as reference points for accuracy–efficiency analysis, subject to access to faithful implementations.

Table 2. Published comparison on the seven-class N-RDD2024 subset

| Model | mAP50 (%) | mAP50–95 (%) | Parameters (M) | GFLOPs |
| --- | --- | --- | --- | --- |
| YOLOv5s | 44.1 | 24.8 | 9.1 | 24.1 |
| YOLOv6s | 41.4 | 23.8 | 16.3 | 44.2 |
| YOLOv8s | 44.1 | 24.9 | 10.8 | 27.3 |
| YOLOv9s | 46.9 | 26.9 | 7.3 | 27.4 |
| YOLOv10s | 43.3 | 25.2 | 8.1 | 24.8 |
| YOLOv11s | 44.6 | 25.7 | 9.4 | 21.6 |
| YOLOv12s | 43.8 | 24.8 | 9.3 | 21.5 |
| YOLO-RD | 45.5 | 26.2 | 6.5 | 24.0 |

Source: selected columns from Table 11 of Wang et al. [3]. Model names are retained as reported. These are published results, not reproductions. Partition: 12,625 training / 1,750 validation / 3,492 test images; seven classes. M = million parameters. GFLOPs are paper-reported values and do not measure complete pipeline latency.

Three evidence boundaries recur. First, improvement on an augmented test set establishes performance against the chosen transformation process; it does not automatically establish resistance to natural camera degradation. Second, a higher clean mAP does not quantify the rate at which performance deteriorates as corruption becomes stronger. Third, a model-level runtime does not measure a complete service that decodes video, processes detections, associates locations, removes duplicates, and uploads records. These are methodological implications of the reviewed evidence, not claims that every paper omits every component.

The scope of the evidence also prevents a global claim that no earlier work has addressed a proposed combination. The present corpus establishes promising directions and specific limitations, but a publication claiming a first-of-its-kind architecture would require a broader, documented novelty search. Negative findings, such as enhancement harming thin cracks or a robust model losing clean-image accuracy, would be informative and should be reported alongside positive averages.

## 6. Research gaps and a controlled evaluation agenda

### 6.1. Generalization beyond familiar corruptions

The clearest shared opportunity is stronger evidence of transfer from modeled degradation to real operating conditions. This follows directly from the synthetic-data limitations in [4,5] and the explicitly untested degradation families in [6]. A useful study would train on specified conditions and reserve at least one degradation family, acquisition device, or route for evaluation. It should distinguish a new corruption severity from a genuinely unseen corruption type and should test realistic compound conditions separately.

Synthetic stress tests remain valuable because they offer repeatable severity control. However, they should complement a separately annotated natural-degradation set. Evaluation should retain difficult images and document exclusions instead of silently removing the failures the system is intended to solve. If paired clean and degraded observations are unavailable, the study should acknowledge the resulting uncertainty in separating acquisition effects from scene differences.

### 6.2. Structure preservation and task-specific robustness

Small defects and narrow cracks create a tension between removing nuisance variation and retaining meaningful high-frequency detail. The resulting hypothesis is that task-aware robustness must preserve evidence for localization, not merely improve visual appearance. Tests should report AP by defect category and, where annotation support permits, by object size and corruption severity. A separate segmentation task could examine continuity or boundary quality, but should not be presented as equivalent to box detection.

Restoration quality measures can supplement this analysis, yet downstream detection must remain the primary endpoint for a detection system. Clean-image performance should be reported beside degraded-image performance to reveal whether robustness is obtained by sacrificing routine operation. Rare-class improvements need supporting object counts and uncertainty estimates; a large percentage change based on very few instances is not strong evidence of a stable benefit.

### 6.3. Temporal reliability and municipal inspection

Guo et al. [5] explicitly identify video continuity as an unresolved challenge. For municipal inspection, the practical unit of interest is a physical defect along a route, rather than every frame containing that defect. A proposed evaluation should therefore distinguish frame-level recognition from event-level reporting. Relevant outcomes include missed physical defects, duplicate reports, temporal inconsistency, and location error, evaluated against a route-level reference inventory.

A camera–location pipeline additionally requires timestamp alignment, image-quality handling, and a rule for retaining representative observations. Tracking can consolidate repeated detections, but cannot recover a defect never reliably observed. These operational proposals extend the review’s analysis; they are not new measured capabilities of the methods in [3–6]. They should be validated under realistic vehicle speed, camera position, and connectivity conditions before claiming an effective municipal service.

### 6.4. A minimum controlled experimental design

The recommended sequence begins with a reproducible detector baseline and a fixed dataset release, class map, and route-separated split where route identifiers are available. The next experiment adds degradation augmentation alone. A third adds a carefully specified learning or enhancement modification, and a fourth combines it with augmentation. This structure helps distinguish the effect of the proposed mechanism from the effect of simply exposing the model to more varied training images. A conventional enhancement method should be included when the contribution concerns restoration.

All variants should share image resolution, initialization policy, training budget, model-selection rule, and inference settings where technically appropriate. Differences that cannot be controlled should be disclosed. Hyperparameters and checkpoint selection belong to the validation set; the final held-out test set should not drive repeated tuning. Multiple random seeds, or a clear limitation when they are infeasible, are necessary to assess whether small gains are repeatable. Confidence intervals should respect correlated observations; route-level resampling is preferable to treating adjacent video frames as independent samples.

Report clean mAP50 and mAP50–95, per-corruption scores, and the absolute drop from clean performance. Include precision, recall, class support, parameter count, memory use, and complete pipeline latency on named hardware. A corruption average should specify its weighting so that common or easy cases cannot silently dominate the conclusion. Tables should distinguish published results from reproduced baselines and new experiments. Exact configurations, split manifests, checkpoint-selection criteria, and code versions are necessary for another group to interpret the comparison.

### 6.5. A defensible contribution statement

A plausible research question is whether a lightweight, structure-preserving detector can improve generalization to unseen combinations of noise and blur while retaining clean accuracy and acceptable end-to-end latency. The contribution would rest on a defined mechanism, controlled comparisons, and realistic evaluation, not on choosing a newer YOLO version or increasing the number of training epochs. If the method uses curriculum learning, attention, enhancement, or distillation, those established components must be attributed to their prior literature.

An empirical contribution may be equally valuable: a well-documented natural-degradation benchmark, route-separated evaluation, or analysis demonstrating when existing robust methods fail could address the evidence gap without claiming an entirely new architecture. The current review supports these as research directions. It does not establish their novelty across all published work or predict a guaranteed numerical improvement.

## 7. Limitations of the review

The selected corpus emphasizes recent journal work relevant to the present problem and is too small and purposively selected to represent the entire road-inspection literature. The subscription-access limitation for [2] restricts its use to thematic context. Published metrics are treated as reported evidence rather than independently reproduced results. Hardware, data selection, and annotation differences prevent quantitative pooling, while incomplete access to exact implementations can constrain reproducibility. The proposed evaluation agenda is a reasoned synthesis and remains to be tested. A later systematic extension should document database searches, eligibility criteria, screening decisions, and a broader range of negative as well as positive findings.

## 8. Conclusion

Recent journal research establishes several credible routes toward road damage recognition under degraded imaging: improved contextual features, expert specialization, progressive supervision, and detection-oriented enhancement. The main unresolved issue is not whether noise matters or whether robustness techniques exist. It is how reliably their benefits transfer across degradation families, acquisition conditions, defect scales, and complete inspection workflows. Dataset and metric differences are substantial enough that isolated accuracy values cannot support a universal ranking.

Future work should prioritize reproducible comparisons that separate clean accuracy from robustness, synthetic stress testing from real-world transfer, and frame-level detection from useful route-level reporting. For a practical road-inspection study, a carefully controlled demonstration of improved performance on unseen noise and blur, with preserved defect structure and measured deployment cost, would provide a stronger contribution than an unsupported claim of novelty based on architectural combination alone.

## References

[1] S. H. Abdelwahed, B. K. Sharobim, B. Wasfey, and L. A. Said, “Advancements in real-time road damage detection: a comprehensive survey of methodologies and datasets,” Journal of Real-Time Image Processing, vol. 22, art. 137, 2025. https://doi.org/10.1007/s11554-025-01683-1

[2] E. E.-D. Hemdan and M. E. Al-Atroush, “A Review Study of Intelligent Road Crack Detection: Algorithms and Systems,” International Journal of Pavement Research and Technology, vol. 19, pp. 1704–1734, 2026 (published online 7 May 2025). https://doi.org/10.1007/s42947-025-00556-x

[3] P. Wang, L. Cheng, J. Liu, H. Wang, D. Wu, G. Ma, and W. Ma, “YOLO-RD: Road defect detection with context-aware attention and balanced loss,” Neurocomputing, vol. 662, art. 132000, 2026. https://doi.org/10.1016/j.neucom.2025.132000

[4] J. Zhao, M. Jia, S. Wang, C. Jiang, D. Niu, and X. Fang, “Robust road damage detection via multi-expert collaboration and cross-scale attention,” Advanced Engineering Informatics, vol. 72, art. 104452, 2026. https://doi.org/10.1016/j.aei.2026.104452

[5] M. Guo, H. Huang, F. Zeng, M. Nie, and M. Dang, “Uncertainty-guided progressive learning for pavement crack segmentation under challenging visual degradation in smartphone-based imaging,” Measurement, vol. 273, art. 121178, 2026. https://doi.org/10.1016/j.measurement.2026.121178

[6] T. Duan, G. Li, H. Zhang, R. Sun, B. Zhang, Y. Xun, J. Sun, and M. Mansurova, “NER: Noise-aware enhancement-assisted real-time detection transformer for pavement damage detection in hazy and low-light environment,” Expert Systems with Applications, vol. 323, art. 132469, 2026. https://doi.org/10.1016/j.eswa.2026.132469

[7] Ö. Kaya and M. Y. Çodur, “N-RDD2024:Road damage and defects,” Mendeley Data, version 3, 2024. Dataset. https://doi.org/10.17632/27c8pwsd6v.3

[8] Y. Liu, J. Yao, X. Lu, R. Xie, and L. Li, “DeepCrack: A deep hierarchical feature learning architecture for crack segmentation,” Neurocomputing, vol. 338, pp. 139–153, 2019. https://doi.org/10.1016/j.neucom.2019.01.036

[9] D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, and Y. Sekimoto, “RDD2022: A multi-national image dataset for automatic road damage detection,” Geoscience Data Journal, vol. 11, pp. 846–862, 2024. https://doi.org/10.1002/gdj3.260
