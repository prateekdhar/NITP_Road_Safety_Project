# Road damage detection presentation

19 slides in LaTeX Beamer (16:9), following the requested section order.

## Use

Open `road_damage_presentation.pdf` in a PDF viewer and select full-screen mode to present.
Edit `road_damage_presentation.tex` to change the content.

To compile using MiKTeX or TeX Live, run twice from this folder:

```sh
pdflatex -interaction=nonstopmode -halt-on-error road_damage_presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error road_damage_presentation.tex
```

For Overleaf, upload the source ZIP, select `road_damage_presentation.tex` as the main document and use pdfLaTeX.

The PDF is a LaTeX slide presentation, not a native PowerPoint file. The source bundle contains the PNG institute logo and four road photographs.

## Evidence boundaries

- Excludes Colab notebook activity, experiments and results.
- Published paper results are explicitly attributed to Zeng and Zhong.
- Local results use the final epoch of `runs/train/first_model2/results.csv` (not independent best-checkpoint evaluation).
- No superiority over the paper, held-out test result, or project publication is claimed.
- Custom variants are labelled PD-inspired, not exact reproductions.
- The uploaded `selected_paper.pdf` is the reference paper. `paper_se.pdf` was empty and could not supply evidence.

## Sources and assets

1. Jiayi Zeng and Han Zhong. YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n model. Scientific Reports 14, 12052 (2024). https://doi.org/10.1038/s41598-024-62933-z
   Paper Tables 1 and 5 provide aggregate model results. Parameter and computation reductions are calculated from rounded table entries. Table values are preferred where the prose differs.
2. RDD2022: https://github.com/sekilab/RoadDamageDetector
   Images copied unchanged from the project's `test_samples/*_1.jpg`. Their dataset origins are retained in `image_sources.json`. Images demonstrate dataset classes; they are not model predictions.
3. Ultralytics pinned implementation: https://github.com/ultralytics/ultralytics/tree/v8.3.203
4. Local evidence: `docs/dataset_audit.json`, `data/rdd/split_manifest.json`, `docs/architecture.md`, `runs/train/first_model2/experiment.json`, `args.yaml`, and `results.csv`. Copies of small audit/experiment files are included in `evidence/`.
5. NIT Patna logo, official website PNG: https://www.nitp.ac.in/_next/static/media/logo.0wkg4jthc83lp.png (retrieved 17 September 2026). Used for institutional identification; no endorsement claim.

## Reporting conventions

mAP and precision/recall are reported as percentages. A change from 69.2% to 70.6% is 1.4 percentage points, not 1.4% relative improvement. Model parameter and GFLOP values shown on the published-results slide belong to the reference paper. The local split is image-based; exact duplicate checks do not establish absence of correlated adjacent frames.
