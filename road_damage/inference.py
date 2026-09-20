"""Shared checkpoint validation and prediction output."""
from pathlib import Path

from ultralytics import YOLO
from road_damage import CLASSES, DESCRIPTIONS


def load_detector(weights):
    path = Path(weights)
    if not path.is_file() or path.suffix != ".pt":
        raise ValueError("Select an existing trained .pt checkpoint, such as runs/train/baseline/weights/best.pt")
    model = YOLO(str(path), task="detect")
    if model.names != CLASSES:
        raise ValueError("This checkpoint is not a four-class RDD detector. Train it on the prepared dataset first.")
    return model


def detections(result):
    rows = []
    for box in result.boxes:
        code = CLASSES[int(box.cls.item())]
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        rows.append(dict(code=code, damage=DESCRIPTIONS[code], confidence=round(float(box.conf.item()), 5),
                         x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2)))
    return rows
