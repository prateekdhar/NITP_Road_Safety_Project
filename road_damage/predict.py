"""Predict an image, image folder, or local video and save annotated results."""
import argparse
import json
from pathlib import Path
from road_damage.inference import load_detector, detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--conf", default=.25, type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        if not args.source.exists():
            raise ValueError(f"Source not found: {args.source}")
        if not 0 <= args.conf <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        model = load_detector(args.weights)
    except ValueError as exc:
        parser.exit(2, f"Prediction setup failed: {exc}\n")
    stream = model.predict(source=str(args.source), conf=args.conf, device=args.device,
                           save=True, stream=True, project="runs/predict", name="road")
    target = None
    try:
        for index, result in enumerate(stream):
            if target is None:
                target = (Path(result.save_dir) / "detections.jsonl").open("w", encoding="utf-8")
            target.write(json.dumps(dict(source=str(result.path), result_index=index,
                                         detections=detections(result))) + "\n")
    finally:
        if target:
            target.close()


if __name__ == "__main__":
    main()
