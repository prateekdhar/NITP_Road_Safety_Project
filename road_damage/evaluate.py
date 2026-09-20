"""Evaluate a trained detector and save aggregate and per-class metrics."""
import argparse
import json
from pathlib import Path
from road_damage import CLASSES
from road_damage.inference import load_detector
from road_damage.train import validate_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/rdd/data.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    try:
        config = validate_data(args.data)
        if not config.get(args.split):
            raise ValueError(f"Dataset has no {args.split} split")
        model = load_detector(args.weights)
    except ValueError as exc:
        parser.exit(2, f"Evaluation setup failed: {exc}\n")
    metrics = model.val(data=str(args.data.resolve()), split=args.split, device=args.device,
                        imgsz=args.imgsz, batch=args.batch, workers=0,
                        project="runs/evaluate", name=args.weights.parent.parent.name + "_" + args.split)
    classes = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        classes[CLASSES[int(class_id)]] = dict(precision=float(p), recall=float(r),
                                              f1=float(2*p*r/(p+r+1e-12)), ap50=float(ap50), ap50_95=float(ap))
    report = dict(weights=str(args.weights.resolve()), split=args.split,
                  metrics={k:float(v) for k,v in metrics.results_dict.items()},
                  per_class=classes, speed_ms=metrics.speed,
                  parameters=sum(p.numel() for p in model.model.parameters()))
    path = Path(metrics.save_dir) / "metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metrics written to {path}")


if __name__ == "__main__":
    main()
