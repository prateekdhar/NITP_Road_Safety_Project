# Restored presentation version

This folder restores the archived YOLO code corresponding to your 19-slide presentation.
The trained model is runs/train/first_model2/weights/best.pt. No retraining was performed.
The original detection and model-statistics pages, dataset tools, training/evaluation
source and PD-inspired modules are restored. Training commands are available as source,
but do not run them unless you explicitly want a new experiment.

## Run on this computer

Open C:\Users\tirth\project3 in VS Code. Double-click run_app.cmd, or run:

```powershell
cd C:\Users\tirth\project3
.\run_app.cmd
```

Open http://localhost:8503. Upload one of the images in test_samples.
Use the sidebar to open Model Statistics. The launcher uses the existing Project
Python environment if this folder has no local .venv.

## Files

- app.py and pages/: original two-page Streamlit interface.
- road_damage/: restored dataset, model, training, evaluation and inference code.
- data/rdd/: original prepared 5,320/665/665 train/validation/test split.
- runs/train/first_model2/: previous weights, training history and plots.
- output/presentation/: the exact supplied PDF and its editable LaTeX source/assets.
- output/road_damage_latex_source.zip: portable source archive.
- RESTORE_MANIFEST.json: backup origins and file checksums.

The saved real-data model is the three-epoch YOLOv8n baseline described in the slides.
PD-inspired architectural modules are implemented, but no trained PD result is claimed.
Historical validation: precision 37.92%, recall 12.30%, mAP@50 8.33%, mAP@50-95 3.16%.
No new accuracy evaluation or training is performed during restoration.

The prior Project and project2 folders have been preserved. The newer SSD work and
the redesigned app/deck were not substituted into this restored version.
