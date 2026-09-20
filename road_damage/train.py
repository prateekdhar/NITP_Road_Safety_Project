"""Train the YOLOv8n baseline or an explicitly paper-inspired experimental variant."""
import argparse
import json
import platform
import subprocess
from pathlib import Path
import sys

import yaml
from ultralytics import YOLO

from road_damage import CLASSES
from road_damage.models import RoadTrainer, VARIANTS


def validate_data(path):
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run python -m road_damage.dataset first.")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset YAML: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a YAML mapping")
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != CLASSES:
        raise ValueError(f"Dataset names must be exactly {CLASSES}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/rdd/data.yaml"))
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    parser.add_argument("--scale", choices=["n", "s", "m"], default="n")
    parser.add_argument("--optimizer", choices=["SGD", "AdamW"], default="SGD")
    parser.add_argument("--lr0", type=float, default=.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA GPU index, e.g. 0")
    parser.add_argument("--workers", type=int, default=0, help="0 is portable on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    parser.add_argument("--scratch", action="store_true", help="Do not download/use pretrained weights")
    parser.add_argument("--resume", type=Path, help="Resume interrupted last.pt using its experiment.json settings")
    args = parser.parse_args()
    resume = args.resume
    if resume:
        try:
            metadata = json.loads((resume.resolve().parent.parent/'experiment.json').read_text())
            for key, value in metadata.items():
                if hasattr(args, key) and key != 'resume':
                    setattr(args, key, value)
            args.data = Path(args.data)
        except (OSError, ValueError) as exc:
            parser.exit(2, f"Cannot resume without valid experiment.json: {exc}\n")
    try:
        validate_data(args.data)
        if args.variant != "baseline" and args.scale != "n":
            raise ValueError("Custom variants support --scale n only")
        if not 0 < args.lr0 <= 1 or args.patience < 0:
            raise ValueError("Learning rate must be in (0,1]; patience must be nonnegative")
        if args.epochs < 1 or args.batch < 1 or args.imgsz < 64 or args.imgsz % 32:
            raise ValueError("Use positive epochs/batch and an image size >=64 divisible by 32")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Training setup failed: {exc}\n")
    model = YOLO(str(resume) if resume else (f"yolov8{args.scale}.yaml" if args.scratch else f"yolov8{args.scale}.pt"))
    trainer = type("ConfiguredRoadTrainer", (RoadTrainer,), {"variant": args.variant, "scale": args.scale})
    def record_start(active):
        output = Path(active.save_dir)
        metadata = {**vars(args), "resume":str(resume) if resume else None, "name":output.name, "data":str(args.data.resolve()),
                    "python":platform.python_version(), "status":"started",
                    "architecture_note":"Baseline or paper-inspired experiment; not an exact paper reproduction"}
        (output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest = args.data.resolve().parent / "split_manifest.json"
        if manifest.is_file():
            import shutil
            shutil.copy2(manifest, output / "dataset_manifest.json")
    model.add_callback("on_train_start", record_start)
    def print_epoch(active):
        print(f"[{Path(active.save_dir).name}] Epoch {active.epoch + 1}/{active.epochs} starting", flush=True)
    model.add_callback("on_train_epoch_start", print_epoch)
    model.train(trainer=trainer, data=str(args.data.resolve()), epochs=args.epochs,
                batch=args.batch, imgsz=args.imgsz, device=args.device, workers=args.workers,
                seed=args.seed, deterministic=True, optimizer=args.optimizer, lr0=args.lr0,
                patience=args.patience, close_mosaic=min(10, args.epochs), project="runs/train",
                name=args.name or (args.variant if args.scale == "n" else f"baseline_{args.scale}"),
                pretrained=not args.scratch, amp=False, plots=True,
                resume=str(resume) if resume else False)
    output = Path(model.trainer.save_dir)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    metadata = {**vars(args), "resume":str(resume) if resume else None, "name":output.name, "status":"completed", "data": str(args.data.resolve()), "python": platform.python_version(),
                "architecture_note": "Baseline or paper-inspired experiment; not an exact paper reproduction"}
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
