"""Sequential 100-epoch detector comparison. Existing runs are never overwritten."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/additional_baselines_20261019'
DATA = ROOT / 'experiments/oct04_200/baseline.yaml'
TRAIN_WORKERS = int(os.environ.get('RDD_TRAIN_WORKERS', '0'))
assert 0 <= TRAIN_WORKERS <= 4
SPECS = {'yolo26n': 'yolo26n.pt', 'yolo26s_100': str(ROOT/'yolo26s.pt')}
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('WANDB_MODE', 'disabled')

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [{k.strip(): v.strip() for k,v in r.items()} for r in csv.DictReader(f)]

def config(key, smoke=False):
    return dict(data=str(DATA), epochs=100, patience=0, imgsz=640, batch=4,
                device=0, workers=TRAIN_WORKERS, amp=True, cache=False, seed=42,
                deterministic=True, optimizer='auto', close_mosaic=10,
                save=True, save_period=10, plots=True, exist_ok=False,
                pretrained=not key.endswith('_scratch'), project=str(EXP/'runs'),
                name=key, **({} if not smoke else {}))

def prepare():
    import ultralytics
    import torch
    assert ultralytics.__version__ == '8.4.161', 'Revalidate if framework changes'
    manifest = ROOT/'data/processed/split_manifest.json'
    assert digest(manifest) == '25a8ab1f25b485b3db90a6bcfd1d9ae6c44ccaa53479e8dc3561780f7b68817e'
    prior = ROOT/'runs/nrdd2024_yolov9s_baseline_100_oct04'
    assert int(float(rows(prior/'results.csv')[-1]['epoch'])) == 100
    assert (prior/'weights/best.pt').is_file()
    EXP.mkdir(parents=True, exist_ok=True)
    plan = dict(created_at=now(), epochs_each=100, queued_epochs=200,
                order=list(SPECS), skipped={'yolov9s': str(prior)},
                blocked={'YOLO-RD': 'Neurocomputing DOI10.1016/j.neucom.2025.132000: exact author implementation not located. Unrelated YOLO-RD repositories must not be substituted.'},
                specs=SPECS, configs={k: config(k) for k in SPECS},
                data=str(DATA), split_sha256=digest(manifest),
                framework=ultralytics.__version__, torch=torch.__version__,
                notes=[
                    'YOLOv5su is Ultralytics updated anchor-free v5, not original anchor-based YOLOv5s.',
                    'YOLOv6s is the bundled Ultralytics YAML implementation, trained from scratch; not an exact Meituan repository reproduction.',
                    'Both models start from pretrained source weights; YOLO26s is a fresh 100-epoch schedule, not a resume of the 300-epoch run.',
                    'Uses our 10-class split, not the paper seven-class split. Not a reproduction of paper scores or training schedule.',
                    'Previous YOLOv9 baseline reused; no extra noise augmentation. Standard framework augmentations retained.',
                    'Exactly 100 epochs per run; best selected on validation. Held-out test untouched.',
                    'Synthetic 5-condition validation follows each run. Smoke checks are excluded from research results.',
                ], sources=['https://github.com/wstchh/RM-RDD','https://doi.org/10.1016/j.neucom.2025.132000',
                            'https://docs.ultralytics.com/models/yolov5','https://docs.ultralytics.com/models/yolov6'])
    if not (EXP/'plan.json').exists():
        save(EXP/'plan.json', plan)
    return json.loads((EXP/'plan.json').read_text())

def initialize(key):
    from ultralytics import YOLO
    source = SPECS[key]
    return YOLO(source if source.endswith('.yaml') else str(ROOT/'weights'/source))

def run_one(key, smoke=False, resume_checkpoint=None):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    from ultralytics.models.yolo.detect import DetectionTrainer
    class EfficientLoaderTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            original_workers = self.args.workers
            try:
                if mode == 'val':
                    self.args.workers = 0
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = original_workers
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA required'
    prepare()
    cfg=config(key)
    if smoke:
        cfg.update(data=str(ROOT/'experiments/paper_noise_20261010/gpu_preflight_20261010_184826/smoke.yaml'),
                   epochs=1, workers=0, project=str(EXP/'preflight'), close_mosaic=0, plots=False, save_period=-1)
    run=Path(cfg['project'])/key
    result_path=EXP/('preflight_status' if smoke else 'status')/(key+'.json')
    if run.exists() and not resume_checkpoint:
        raise RuntimeError(f'Refusing to overwrite {run}; use checkpoint resume explicitly')
    model=YOLO(resume_checkpoint) if resume_checkpoint else initialize(key)
    start_epoch = 0
    if resume_checkpoint:
        assert not smoke and model.ckpt.get('optimizer') is not None
        start_epoch = model.ckpt['epoch'] + 1
        assert 0 < start_epoch < 100 and model.ckpt['train_args']['epochs'] == 100
        assert int(float(rows(run/'results.csv')[-1]['epoch'])) == start_epoch
    source=SPECS[key]
    source_path=ROOT/'weights'/source
    metadata={'source':source, 'pretrained':cfg['pretrained'], 'started_at':now(), 'smoke_only':smoke}
    if resume_checkpoint:
        metadata.update(resumed_from_epoch=start_epoch,resume_checkpoint=resume_checkpoint,
                        resume_sha256=digest(resume_checkpoint),
                        loader_note=f'{TRAIN_WORKERS} training workers, 0 validation workers; resume may change RNG trajectory')
    if source_path.is_file(): metadata['source_sha256']=digest(source_path)
    def progress(t):
        csv_path=Path(t.save_dir)/'results.csv'
        completed=int(float(rows(csv_path)[-1]['epoch'])) if csv_path.exists() else 0
        save(result_path, dict(metadata,state='training',epoch_completed=completed,epochs=cfg['epochs'],
                              pid=os.getpid(),updated_at=now(),run=str(t.save_dir),
                              metrics={k:float(v) for k,v in t.metrics.items()}))
    def verify(t):
        assert t.epochs==cfg['epochs'] and t.data['nc']==10
        assert t.start_epoch == start_epoch
        assert Path(t.save_dir).resolve()==run.resolve()
        metadata.update(parameters=sum(p.numel() for p in t.model.parameters()),
                        gflops_640=get_flops(t.model,640), optimizer=type(t.optimizer).__name__)
        save(run/('resume_provenance.json' if resume_checkpoint else 'provenance.json'), dict(metadata,config=cfg))
    model.add_callback('on_fit_epoch_end',progress)
    model.add_callback('on_train_start',verify)
    if resume_checkpoint:
        model.train(trainer=EfficientLoaderTrainer,resume=True,device=0,workers=TRAIN_WORKERS,
                    save_dir=str(run),cache=False)
    else:
        model.train(trainer=EfficientLoaderTrainer,**cfg)
    history=rows(run/'results.csv')
    assert int(float(history[-1]['epoch']))==cfg['epochs']
    assert len(history)==cfg['epochs']
    best=run/'weights/best.pt'
    assert best.is_file()
    best_epoch=max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))
    report=dict(metadata,state='complete',epochs_completed=cfg['epochs'],finished_at=now(),
                best=str(best),sha256=digest(best),best_validation_epoch=best_epoch,
                latest_validation_epoch=history[-1],run=str(run))
    if not smoke:
        metrics=YOLO(str(best)).val(data=str(DATA),split='val',imgsz=640,batch=4,device=0,
                                   workers=0,plots=True,project=str(EXP/'validation'),name=key)
        report['validation']={'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
                              'speed':metrics.speed,'per_class':[
                                  dict(class_id=int(c),precision=float(metrics.box.p[i]),recall=float(metrics.box.r[i]),
                                       ap50=float(metrics.box.ap50[i]),ap50_95=float(metrics.box.ap[i]))
                                  for i,c in enumerate(metrics.box.ap_class_index)]}
    save(result_path,report)

def queue(preflight_only=False):
    prepare()
    lock=EXP/'active_queue.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    failures={}
    def execute(key,args,phase):
        log=EXP/'logs'/f'{key}_{phase}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf-8') as f:
            p=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            save(EXP/'queue_status.json',dict(state=phase,current=key,queue_pid=os.getpid(),
                                             child_pid=p.pid,log=str(log),updated_at=now(),failures=failures))
            code=p.wait()
        if code:
            failures[key]=dict(phase=phase,exit_code=code,log=str(log))
        return code==0
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        if preflight_only:
            for key in SPECS:
                receipt=EXP/'preflight_status'/f'{key}.json'
                if receipt.exists() and json.loads(receipt.read_text()).get('state')=='complete':continue
                execute(key,[str(Path(__file__)),'--smoke',key],'preflight')
        else:
            for key in SPECS:
                preflight=EXP/'preflight_status'/f'{key}.json'
                if not preflight.exists() or json.loads(preflight.read_text()).get('state')!='complete':
                    failures[key]={'phase':'preflight','reason':'No successful preflight'}
                    continue
                receipt=EXP/'status'/f'{key}.json'
                if receipt.exists() and json.loads(receipt.read_text()).get('state')=='complete':
                    completed=json.loads(receipt.read_text())
                    matches=[]
                    for candidate in (ROOT/'experiments/paper_noise_20261010').glob('evaluation_*/comparison.json'):
                        saved=json.loads(candidate.read_text())
                        entry=saved.get('models',{}).get(key,{})
                        if entry.get('sha256')==completed['sha256']: matches.append(candidate)
                    if matches:
                        candidate=max(matches,key=lambda p:p.stat().st_mtime)
                        saved=json.loads(candidate.read_text())
                        done=candidate.parent/'status.json'
                        if done.exists() and json.loads(done.read_text()).get('state')=='complete' and len(saved['models'][key]['conditions'])==5:continue
                        if not execute(key,[str(ROOT/'evaluate_paper_study.py'),'--device','0','--batch','2','--weights',key+'='+completed['best'],'--resume-dir',str(candidate.parent)],'stress_validation'):raise RuntimeError('Interrupted evaluation could not be completed')
                    else:
                        if not execute(key,[str(ROOT/'evaluate_paper_study.py'),'--device','0','--batch','2','--weights',key+'='+completed['best']],'stress_validation'):raise RuntimeError('Evaluation failed')
                    continue
                command=[str(Path(__file__)),'--run',key]
                requests=EXP/'resume_requests.json'
                if requests.exists():
                    checkpoint=json.loads(requests.read_text()).get(key)
                    if checkpoint: command.extend(['--resume-checkpoint',checkpoint])
                if not execute(key,command,'training'):continue
                best=json.loads(receipt.read_text())['best']
                execute(key,[str(ROOT/'evaluate_paper_study.py'),'--device','0','--batch','2','--weights',key+'='+best],'stress_validation')
        save(EXP/('preflight_summary.json' if preflight_only else 'queue_summary.json'),dict(failures=failures,finished_at=now()))
        save(EXP/'queue_status.json',dict(state=('preflight_complete' if preflight_only else 'available_models_complete') if not failures else 'needs_attention',
                                        failures=failures,blocked=['YOLO-RD'],updated_at=now()))
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--queue',action='store_true')
    parser.add_argument('--smoke',choices=SPECS)
    parser.add_argument('--run',choices=SPECS)
    parser.add_argument('--resume-checkpoint')
    args=parser.parse_args()
    if args.all:
        queue(True)
        assert not json.loads((EXP/'preflight_summary.json').read_text())['failures'], 'Preflight failed; inspect logs'
        queue()
    elif args.smoke:run_one(args.smoke,True)
    elif args.run:run_one(args.run,resume_checkpoint=args.resume_checkpoint)
    elif args.preflight:queue(True)
    elif args.queue:queue()
    elif args.prepare:print(json.dumps(prepare(),indent=2))
    else:parser.error('Select an operation')
