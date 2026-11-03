"""Isolated smoke tests or explicitly requested fine-tuning; no automatic queue."""
import argparse, functools, hashlib, json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
os.environ.setdefault('WANDB_MODE','disabled')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(d,indent=2),encoding='utf-8'); tmp.replace(p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['augmentation','consistency','proposed'],default='proposed')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-imgsz',type=int,default=160)
    p.add_argument('--smoke-batch',type=int,default=1)
    p.add_argument('--epochs',type=int)
    p.add_argument('--device',default='0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--name')
    a=p.parse_args()
    if not a.smoke and (a.epochs is None or a.epochs<1): p.error('Explicit --epochs required for full-data training')
    import torch, yaml
    from ultralytics import YOLO
    from paper_robustness import PaperTrainer, Method
    from dataclasses import asdict
    torch.set_num_threads(2)
    student=ROOT/'runs/nrdd2024_yolov9s_noise_100_oct04/weights/best.pt'
    teacher=ROOT/'runs/nrdd2024_yolov9s_baseline_100_oct04/weights/best.pt'
    expected={student:'18c5796b9f03c3440ec5f3fabd19bf2e63a1fd53e12612b1e087c03a96c3069b',teacher:'4e3731971fc26b01efca24cf69efd6a641a3c0a80d6b34a4d9303a8a0f0c1569'}
    for path,h in expected.items():
        if digest(path)!=h: raise RuntimeError('Reference checkpoint changed: '+str(path))
    name=a.name or f"paper_{a.variant}_{'smoke' if a.smoke else str(a.epochs)+'ep'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path(name).name!=name or '..' in name: raise ValueError('Name must be a simple folder name')
    out=ROOT/'experiments/paper_noise_20261010'/name
    if out.exists(): raise FileExistsError('Preserving existing run: '+str(out))
    out.mkdir(parents=True)
    data=ROOT/'experiments/oct04_200/baseline.yaml'
    if a.smoke:
        # Copy training-only samples; never use held-out test for implementation tests.
        original=ROOT/'data/processed'
        manifest=json.loads((original/'split_manifest.json').read_text())
        selected=[r for r in manifest if r['split']=='train' and r['labels']][:8]
        assert len(selected)==8
        sample=out/'smoke_data'
        for split in ['train','val']:
            (sample/'images'/split).mkdir(parents=True)
            (sample/'labels'/split).mkdir(parents=True)
            for r in selected[:8 if split=='train' else 2]:
                source=original/r['output']
                shutil.copy2(source,sample/'images'/split/source.name)
                (sample/'labels'/split/(source.stem+'.txt')).write_text('\n'.join(r['labels'])+'\n')
        data=out/'smoke.yaml'
        data.write_text(yaml.safe_dump({'path':str(sample),'train':'images/train','val':'images/val','names':{i:f'D{i*10:02d}' for i in range(10)}}))
    method=asdict(Method(variant=a.variant,teacher=str(teacher)))
    cfg=dict(data=str(data),epochs=1 if a.smoke else a.epochs,imgsz=a.smoke_imgsz if a.smoke else 640,batch=a.smoke_batch if a.smoke else a.batch,
             device=a.device,workers=0,seed=a.seed,deterministic=True,amp=False if a.device=='cpu' else True,
             optimizer='AdamW',lr0=.0001,lrf=.1,weight_decay=.0005,warmup_epochs=0 if a.smoke else 1,
             patience=0,save=True,save_period=10,cache=False,plots=False,
             mosaic=0,close_mosaic=0,mixup=0,cutmix=0,copy_paste=0,multi_scale=0,
             project=str(out),name='train',exist_ok=False)
    save(out/'configuration.json',{'student':str(student),'teacher':str(teacher),'hashes':{str(k):v for k,v in expected.items()},'method':method,'train':cfg,
                                  'smoke_only':a.smoke,'note':'Smoke validation reuses training images; metrics are not research results.'})
    def progress(t):
        save(out/'status.json',{'state':'training','epoch_completed':min(t.epoch+1,t.epochs),'epochs':t.epochs,'pid':os.getpid(),
                               'updated_at':datetime.now(timezone.utc).isoformat(),'metrics':{k:float(v) for k,v in t.metrics.items()}})
    save(out/'status.json',{'state':'starting','pid':os.getpid()})
    try:
        model=YOLO(str(student)); model.add_callback('on_fit_epoch_end',progress)
        model.train(trainer=functools.partial(PaperTrainer,method_config=method),**cfg)
        best=Path(model.trainer.best)
        # Verify deployment checkpoint is a stock detector without a teacher.
        loaded=YOLO(str(best))
        assert not hasattr(loaded.model,'teacher')
        assert type(loaded.model).__module__=='ultralytics.nn.tasks'
        loaded.model.float().eval().to('cpu')
        with torch.no_grad():
            pred=loaded.model(torch.zeros(1,3,160,160))
        assert torch.isfinite(pred[0]).all()
        for path,h in expected.items(): assert digest(path)==h
        save(out/'status.json',{'state':'complete','smoke_only':a.smoke,'epochs_completed':cfg['epochs'],'best':str(best),'sha256':digest(best),
                               'stock_checkpoint_reload_passed':True,'source_checkpoints_unchanged':True,'finished_at':datetime.now(timezone.utc).isoformat()})
        print('STUDY_OUTPUT='+str(out),flush=True)
    except Exception as e:
        save(out/'status.json',{'state':'failed','error':repr(e),'smoke_only':a.smoke})
        raise

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
