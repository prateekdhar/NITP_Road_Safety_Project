import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.ultralytics'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Separate one-epoch GPU validation run')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: training will not silently fall back to CPU')
    config = json.loads((ROOT/'train-config.json').read_text())
    config['data'] = str(ROOT/'data/processed/dataset.yaml')
    config['project'] = str(ROOT/'runs')
    if not Path(config['data']).is_file():
        raise RuntimeError('Run prepare_dataset.py first')
    if args.smoke:
        config.update(epochs=1, fraction=0.02, name='gpu_smoke', workers=0, plots=False)
    model_name = config.pop('model')
    if args.resume:
        model = YOLO(str(args.resume.resolve()))
    else:
        model = YOLO(model_name)
    def progress(trainer):
        state = dict(updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), epoch_completed=trainer.epoch+1, epochs=trainer.epochs, run=str(trainer.save_dir), metrics={k:float(v) for k,v in trainer.metrics.items()})
        path = Path(trainer.save_dir)/'progress.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2))
        temp.replace(path)
    model.add_callback('on_fit_epoch_end', progress)
    if args.resume:
        model.train(resume=True, device=0, data=config['data'],
                    save_dir=str(args.resume.resolve().parent.parent))
    else:
        model.train(**config)
    if not args.smoke:
        from evaluate import evaluate
        evaluate(Path(model.trainer.best), 'test')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
