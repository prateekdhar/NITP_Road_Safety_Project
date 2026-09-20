# Project audit — 17 September 2026

## Result

The automated checks and full prepared-dataset audit passed. This is evidence for the
tested workflows, not a guarantee that every possible input, device or deployment is error-free.

## Corrections made

- Detection now resolves relative checkpoint paths from the project root, so launching
  the app from another working directory does not hide valid models.
- Automatic checkpoint selection uses modification time instead of alphabetic ordering.
- Dataset conversion rejects NaN/infinite coordinates and split ratios, and reports missing
  coordinate fields as validation errors instead of uncaught conversion exceptions.
- Training configuration validation handles empty, malformed and non-mapping YAML.
- The statistics reader excludes non-finite values and invalid epoch numbers; malformed
  experiment metadata produces an explicit error.
- Streamlit is pinned to the verified 1.63.0 release; pandas is an explicit dependency.
- README layout reflects registered navigation and separate detection/statistics pages.
- Raw dataset and copied test samples are excluded from version control.

The previously repaired explicit page registration remains in place.

## Verification performed

- Full expanded suite: **33 passed**, including both page routes, navigation back to
  detection, checkpoint inspection, malformed inputs, and a real image upload/prediction
  from a different working directory using `first_model2/weights/best.pt`.
- Model tests were then strengthened to use the actual Ultralytics detection loss;
  all **8 model tests passed**, covering the baseline and five experimental variants,
  training gradients, inference decoding, custom checkpoint reload and COCO rejection.
- `pip check`: no broken dependency requirements.
- Python compilation passed for the entry point, pages, package, scripts and tests.
- All **6,650 prepared images** were decoded and hashed. Every corresponding label file
  was validated and compared to its original XML annotation. Manifest assignments and
  per-class totals matched. No duplicate decoded images or audit errors were found.
- Dataset counts: 5,320 training, 665 validation, 665 test images.

Machine-readable dataset evidence is in [dataset_audit.json](dataset_audit.json).
The read-only scan can be repeated with `python -m scripts.audit_dataset` while the
original `RDD2022` directory is present in this project.

## Remaining limitations

- The available real checkpoint was trained for only three epochs. Its latest logged
  validation mAP@50 is 8.328%; passing software tests does not establish useful detection accuracy.
- D10 is sparse: 50 training, 9 validation and 4 test annotations. Per-class results will
  be unstable and should not be interpreted as strong generalization evidence.
- The split was created without road-sequence grouping. Exact image duplicates are absent,
  but related frames or nearby road scenes may still cross splits.
- Experimental PD modules are documented approximations of the paper, not a verified
  reproduction. Published 70.6% mAP@50 and 111.9 FPS are not local measurements.
- GPU execution, long training runs, embedded deployment and all possible video formats
  were not verified in this audit. No retraining or fresh full-dataset model evaluation
  was performed; the audit did not alter existing weights or prepared splits.
- The statistics page intentionally shows latest-epoch validation logs. Those can differ
  slightly from the final best-checkpoint evaluation displayed by the training terminal.
