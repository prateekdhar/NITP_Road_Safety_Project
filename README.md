# Restored project3

Start with [START_HERE.md](START_HERE.md). The original project documentation follows; running the app uses the saved checkpoint and does not train a model.

# Road Damage Detection and Classification

A VS Code-ready Python project that detects **D00 longitudinal cracks, D10 transverse cracks,
D20 alligator cracks, and D40 potholes** using YOLOv8n. Includes dataset preparation,
training, evaluation, image/video inference, and a Streamlit image inspection app.

## What is implemented

- A pretrained YOLOv8n transfer-learning baseline.
- Optional `bot`, `lska`, `ghost`, `lscd`, and combined `pd` experiments.
- Strict Pascal VOC annotation conversion to YOLO labels.
- Reproducible train/validation/test splits, exact pixel duplicate removal, and optional sequence grouping.
- Aggregate and per-class evaluation reports; annotated predictions and JSON output.
- VS Code debugging configurations and automated tests.

**The experimental `pd` model is inspired by the paper, not a verified reproduction.**
Its attention and head details are explicit design choices documented in
[docs/architecture.md](docs/architecture.md). No claim is made that it reproduces the paper's
parameter count or accuracy. Real training requires RDD2022, which is not bundled.
Synthetic smoke-test checkpoints are for software verification only.

## Open in VS Code

Open this folder with **File > Open Folder**. Install the recommended Python and Python
Debugger extensions if VS Code prompts. Select `.venv/Scripts/python.exe` through
**Python: Select Interpreter**. Press **F5** and choose **Road Damage: App** to run the interface.

The project environment has been created locally in `.venv`. For a new machine, install
Python 3.11 or 3.12 and run these commands from a PowerShell terminal in the project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`requirements-lock.txt` records the exact versions installed and verified on this Windows
Python 3.12 CPU environment. Use it instead of `requirements.txt` when recreating that setup.

Using the environment's executable directly avoids PowerShell activation-policy issues.
Open the local address printed by Streamlit. The app explains the next steps when no trained
checkpoint is available. It does not use a general COCO model to invent road-damage predictions.

CPU execution works by default. For GPU training, install a matching CUDA-enabled
PyTorch/torchvision pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/),
verify `torch.cuda.is_available()`, and use `--device 0`. Full training on CPU can be slow.

## 1. Download and prepare data

Download annotated RDD2022 archives from the [official dataset repository](https://github.com/sekilab/RoadDamageDetector)
or its [Figshare release](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547).
Extract the images and XML annotations into a directory such as `data/raw/RDD2022`.
The converter searches recursively, so country folders may remain intact. Use annotated
training archives; images without XML annotations are not included.

```powershell
.\.venv\Scripts\python.exe -m road_damage.dataset --source "data/raw/RDD2022" --output "data/rdd"
```

Outputs: `data/rdd/data.yaml`, `images/{train,val,test}`, `labels/{train,val,test}`, and
`split_manifest.json`, which records every split assignment and per-class object counts.
Default ratios are 80/10/10. Images containing labels outside D00/D10/D20/D40 are skipped
in full by default, including mixed images with both target and non-target labels. This
avoids training against partially annotated damage. The manifest records skipped XML paths,
extra label counts, and the selected policy. Extra labels are never remapped to potholes.
Use `--unsupported-classes error` for strict rejection instead. Review skipped annotations,
especially unexpected labels that could be annotation typos, before research training.
The converter refuses nonempty output directories, invalid
boxes, conflicting duplicate annotations, and ambiguous image matches.
VOC boxes are interpreted as 1-based inclusive coordinates, with RDD's observed zero-valued
edge minima also accepted as the image boundary. Negative coordinates, reversed boxes,
and maxima beyond the image dimensions remain errors.

Exact pixel duplicates are removed before splitting. Neighboring frames can still leak
information across splits unless sequence identifiers are supplied. To keep road sequences
together, prepare a CSV with these columns:

```csv
image,group
India/train/JPEGImages/India_000001.jpg,india_route_01
India/train/JPEGImages/India_000002.jpg,india_route_01
```

Every retained annotated image must have an entry. Use forward-slash paths relative to
the source directory, and meaningful route/sequence identifiers. Then add `--groups-csv groups.csv`.
Ratios are allocated by group count, so image counts can differ from the requested percentages.
Splits are not class-stratified; inspect class counts before training and use a sufficiently large dataset.

## 2. Train the baseline

First perform a short real-data run, then use 300 epochs for the full experiment:

```powershell
.\.venv\Scripts\python.exe -m road_damage.train --epochs 3 --batch 4 --name baseline_check
.\.venv\Scripts\python.exe -m road_damage.train --variant baseline --epochs 300 --batch 8 --device 0
```

Use `--device cpu` if CUDA is unavailable. Pretrained `yolov8n.pt` is downloaded on the
first training run. Add `--scratch` for offline random initialization. The data is validated
before model weights are loaded or downloaded.

Defaults: 640-pixel input, SGD, learning rate 0.01, seed 42, Windows-compatible workers=0,
and mixed precision disabled for consistent custom-module behavior. Mosaic augmentation
is disabled for the final 10 epochs. The paper's description of mosaic timing is ambiguous;
this project follows the standard Ultralytics `close_mosaic` convention.

The best model is saved under `runs/train/<name>/weights/best.pt`. Ultralytics increments
the output name if it exists. Each completed experiment also saves its arguments,
`experiment.json`, `environment.txt`, training curves, and metrics.

## 3. Run the paper-inspired experiments

For the expanded GPU experiment workflow, open [gpu_training/START_HERE.md](gpu_training/START_HERE.md).
It includes a Colab notebook and a source bundle with multi-country dataset download,
YOLOv8n/s/m comparisons, validation-only selection and final test evaluation.
Baseline training supports `--scale n`, `--scale s` and `--scale m`. Custom modules remain
restricted to nano. Interrupted runs can resume with `--resume runs/train/<run>/weights/last.pt`;
settings are read from the adjacent `experiment.json`, which is now saved at training start.

```powershell
.\.venv\Scripts\python.exe -m road_damage.train --variant bot --device 0
.\.venv\Scripts\python.exe -m road_damage.train --variant lska --device 0
.\.venv\Scripts\python.exe -m road_damage.train --variant ghost --device 0
.\.venv\Scripts\python.exe -m road_damage.train --variant lscd --device 0
.\.venv\Scripts\python.exe -m road_damage.train --variant pd --device 0
```

Keep data splits, seeds, image size, epochs, and hardware constant. New modules are randomly
initialized; compatible pretrained parameters are transferred. Architecture modifications
can change the amount of transferred initialization, so include scratch-training comparisons
when making research claims. Custom checkpoints reference `road_damage.models`; keep this
package available when loading them on another machine.

## 4. Evaluate

Use validation results to select a model. Run the untouched test split after that selection:

```powershell
.\.venv\Scripts\python.exe -m road_damage.evaluate --weights "runs/train/baseline/weights/best.pt" --split val
.\.venv\Scripts\python.exe -m road_damage.evaluate --weights "runs/train/baseline/weights/best.pt" --split test
```

`runs/evaluate/<experiment>_<split>/metrics.json` contains mAP@50, mAP@50-95,
precision/recall, per-class F1, parameter count, and timing information. Metrics are on a
0–1 scale. Absent ground-truth classes are omitted from the per-class report. Ultralytics
also saves plots and confusion matrices. Timings depend on hardware, batch size, and input
resolution; they are not an end-to-end application FPS benchmark.

## 5. Detect and download results

Use **Model Statistics** in the app sidebar (or **View model statistics and paper results**
on the home page) to view the paper's reported results, architectural characteristics,
and your own saved training curves. The page identifies paper sources, distinguishes
latest-epoch validation results from checkpoint/test evaluation, and offers a JSON download.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m road_damage.predict --weights "runs/train/baseline/weights/best.pt" --source "road.jpg"
.\.venv\Scripts\python.exe -m road_damage.predict --weights "runs/train/baseline/weights/best.pt" --source "road_video.mp4"
```

In the app, select a trained checkpoint, upload an image, set confidence, and click
**Detect road damage**. Download the annotated PNG or JSON report. The UI uses CPU inference.
The CLI accepts images, folders, and local videos, and streams video results without
accumulating all frames in memory. Results go to a unique `runs/predict/road*` directory.
Video counts are per frame; tracking distinct physical defects is not implemented.
Bounding boxes locate damage; they do not estimate depth, severity, or repair cost.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke_train --variant baseline
.\.venv\Scripts\python.exe -m scripts.smoke_train --variant pd
```

The smoke commands create synthetic images under `tmp/smoke` and run one epoch at 64 pixels.
They verify training and checkpoint plumbing, not model quality. Do not use their checkpoints
as a road detector. The app excludes `smoke_*` checkpoints from automatic selection.

Local verification completed: 16 automated tests passed; the combined PD-inspired model
completed one synthetic training epoch, saved and reloaded a checkpoint, evaluated the
synthetic test split, and exported an annotated image plus JSON predictions. These checks
establish software functionality only. No real-data accuracy result is available yet.

## Project layout

```text
app.py                    Registered navigation entry point
pages/0_Detection.py       Image inspection interface
pages/1_Model_Statistics.py Paper results and local training dashboard
road_damage/dataset.py    VOC conversion and split manifest
road_damage/models.py     Custom modules and architecture construction
road_damage/train.py      Training and reproducibility metadata
road_damage/evaluate.py   Validation/test metrics
road_damage/predict.py    Image/video command-line inference
road_damage/inference.py  Checkpoint validation and result conversion
tests/                    Dataset, model, and interface tests
scripts/smoke_train.py    Synthetic end-to-end training check
.vscode/                  Interpreter and debugging configuration
```

## Sources and scope

- Zeng & Zhong (2024), [YOLOv8-PD](https://doi.org/10.1038/s41598-024-62933-z).
- [Ultralytics Python API](https://docs.ultralytics.com/usage/python/) and
  [training settings](https://docs.ultralytics.com/modes/train/).
- [RDD2022 dataset](https://github.com/sekilab/RoadDamageDetector).

Check dependency and dataset licenses before redistribution; Ultralytics is offered under
AGPL-3.0 and enterprise licensing. This project does not bundle author-provided YOLOv8-PD code.

# NITP_Road_Safety_Project
