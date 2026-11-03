import argparse
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.ultralytics'))

def evaluate(weights, split):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    metrics = model.val(data=str(ROOT/'data/processed/dataset.yaml'), split=split, device=0, batch=4, workers=0, imgsz=640, project=str(ROOT/'runs/evaluation'), name=split, plots=True)
    report = dict(split=split, weights=str(weights), metrics={k:float(v) for k,v in metrics.results_dict.items()}, per_class=[])
    for i,c in enumerate(metrics.box.ap_class_index):
        report['per_class'].append(dict(name=model.names[int(c)], precision=float(metrics.box.p[i]), recall=float(metrics.box.r[i]), f1=float(metrics.box.f1[i]), map50=float(metrics.box.ap50[i]), map50_95=float(metrics.box.ap[i])))
    report['note'] = 'Precision/recall/F1 are Ultralytics confidence-curve summaries; test is an internal holdout, not the official challenge benchmark.'
    (Path(metrics.save_dir)/'metrics.json').write_text(json.dumps(report,indent=2))
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--split',choices=['val','test'],default='test')
    a=p.parse_args()
    evaluate(a.weights,a.split)
