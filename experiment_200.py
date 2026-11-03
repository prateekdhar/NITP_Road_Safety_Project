"""Two isolated experiments; exactly 100 epoch ceilings each, validation selection only."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/oct04_200'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('WANDB_MODE', 'disabled')

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)

def prepare():
    import yaml
    EXP.mkdir(parents=True, exist_ok=True)
    if (EXP/'plan.json').exists():
        existing=json.loads((EXP/'plan.json').read_text())
        if existing.get('revision') in (2,3): return existing
        if (EXP/'queue_status.json').exists(): raise RuntimeError('Cannot revise a started queue')
        save(EXP/'plan_provisional_not_run.json',existing)
    data = ROOT/'data/processed'
    manifest = json.loads((data/'split_manifest.json').read_text())
    train = [r for r in manifest if r['split']=='train']
    val = {r['output'] for r in manifest if r['split']=='val'}
    test = {r['output'] for r in manifest if r['split']=='test'}
    assert len(train)==13782 and len(val)==3704 and len(test)==1509
    classes = [{int(line.split()[0]) for line in r['labels']} for r in train]
    counts = Counter(c for cs in classes for c in cs)
    paths=[]; repetitions=Counter(); effective=Counter()
    for r,cs in zip(train,classes):
        assert r['output'] not in val|test
        p=data/r['output']; assert p.is_file()
        # Repeat only rare-class images, at most three times; no test/val statistics used.
        factor=min(3,max(1,math.ceil(max((math.sqrt(.01*len(train)/counts[c]) for c in cs),default=1))))
        paths.extend([p.as_posix()]*factor); repetitions[factor]+=1
        effective.update({c:factor for c in cs})
    (EXP/'train_balanced.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    base={'path':data.as_posix(),'val':'images/val','test':'images/test','names':{i:f'D{i*10:02d}' for i in range(10)}}
    for name,training in [('baseline','images/train'),('balanced',(EXP/'train_balanced.txt').as_posix())]:
        (EXP/f'{name}.yaml').write_text(yaml.safe_dump(dict(base,train=training)),encoding='utf-8')
    common=dict(epochs=100,patience=0,batch=4,device=0,workers=2,amp=True,cache=False,seed=42,deterministic=True,save=True,save_period=10,plots=True,exist_ok=False,project=str(ROOT/'runs'),close_mosaic=10)
    configs={
        'yolov9s_baseline':dict(common,model=str(ROOT/'weights/yolov9s.pt'),data=str(EXP/'baseline.yaml'),imgsz=640,optimizer='auto',name='nrdd2024_yolov9s_baseline_100_oct04'),
        'yolo26s_balanced':dict(common,model=str(ROOT/'yolo26s.pt'),data=str(EXP/'baseline.yaml'),imgsz=640,optimizer='auto',name='nrdd2024_yolo26s_srwbce_100_oct04')}
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),total_epoch_ceiling=200,configs=configs,original_split_sha256=hashlib.sha256((data/'split_manifest.json').read_bytes()).hexdigest(),train_unique=len(train),train_balanced_entries=len(paths),repeat_histogram=dict(repetitions),class_image_counts=dict(counts),effective_class_image_counts=dict(effective),notes=['Both start from COCO pretrained weights, not the previous fitted checkpoint.','No test evaluation during experiments. Compare validation at640 for both, plus native832 for enhanced model.','Combined recipe experiment, not a causal ablation or a reproduction of custom YOLO-RD.','Exactly100epochs per model; retain best validation checkpoint. Oversampling changes examples per epoch and832 changes compute.'])
    box_counts=Counter(int(line.split()[0]) for r in train for line in r['labels'])
    plan.update(revision=2,srwbce_weights=[(max(box_counts.values())/box_counts[i])**.25 for i in range(10)],srwbce_n=4,source='YOLO-RD, Neurocomputing662(2026)132000 DOI10.1016/j.neucom.2025.132000 Eq17 and section4.4.3',notes=['YOLOv9s baseline and YOLO26s with adapted SR-WBCE,100epochs each.','Both use original training split,640pixels,default optimizer selection; provisional repeated manifest is NOT used.','Weights use training box counts only: (max(N)/N_class)^(1/4), applied to positive AND negative BCE terms per Eq17. No normalization or clipping.','SR-WBCE was studied on YOLOv8s; adaptation to both YOLO26 heads is experimental, not full YOLO-RD reproduction.','YOLO26 architecture reference: Applied Sciences16(13)6758(2026), DOI10.3390/app16136758; primarily nano/COCO evidence, not RDD2024 proof.','No test-set tuning or evaluation in this queue. Different model families prevent causal attribution of any gain solely to loss.','Original300epoch run remains untouched.'])
    save(EXP/'plan.json',plan)
    return plan

def run_one(key):
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available(), 'CUDA is required'
    plan=prepare(); cfg=plan['configs'][key].copy(); weights=cfg.pop('model')
    model=YOLO(weights)
    robust=plan.get('revision')==3 and key=='yolo26s_balanced'
    if key=='yolo26s_balanced' and not robust:
        def install_loss(t):
            criterion=t.model.init_criterion()
            heads=[criterion.one2many,criterion.one2one]
            for head in heads:
                head.bce.weight=torch.tensor(plan['srwbce_weights'],device=head.device,dtype=torch.float32)
            t.model.criterion=criterion
            print('Installed SR-WBCE n=4 on BOTH heads. Weights:',plan['srwbce_weights'],flush=True)
        model.add_callback('on_pretrain_routine_end',install_loss)
    def progress(t):
        save(EXP/f'{key}_progress.json',dict(epoch_completed=min(t.epoch+1,t.epochs),epochs=t.epochs,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),metrics={k:float(v) for k,v in t.metrics.items()}))
    model.add_callback('on_fit_epoch_end',progress)
    if robust:
        from noise_training import NoiseTrainer
        model.train(trainer=NoiseTrainer,**cfg)
    else:
        model.train(**cfg)
    best=Path(model.trainer.best)
    report={'best':str(best),'best_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),'validation':{}}
    for size in sorted({640,cfg['imgsz']}):
        metrics=YOLO(str(best)).val(data=str(EXP/'baseline.yaml'),split='val',imgsz=size,batch=4,workers=0,device=0,project=str(EXP/'validation'),name=f'{key}_{size}',plots=True)
        report['validation'][str(size)]={'metrics':{k:float(v) for k,v in metrics.results_dict.items()},'per_class':[dict(class_id=int(c),ap50=float(metrics.box.ap50[i]),ap50_95=float(metrics.box.ap[i]),precision=float(metrics.box.p[i]),recall=float(metrics.box.r[i])) for i,c in enumerate(metrics.box.ap_class_index)]}
    save(EXP/f'{key}_complete.json',report)
    if robust:
        from noise_training import validate_robustness
        baseline=json.loads((EXP/'yolov9s_baseline_complete.json').read_text())
        validate_robustness(ROOT,EXP,{'baseline':baseline['best'],'noise_robust':str(best)})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',choices=['yolov9s_baseline','yolo26s_balanced']); p.add_argument('--queue',action='store_true'); a=p.parse_args()
    plan=prepare()
    if a.prepare:
        print(json.dumps(plan,indent=2)); return
    if a.run:
        run_one(a.run); return
    if not a.queue: p.error('Choose --prepare, --run or --queue')
    lock=EXP/'queue.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        for key in plan['configs']:
            if (EXP/f'{key}_complete.json').exists(): continue
            save(EXP/'queue_status.json',dict(state='running',experiment=key,pid=os.getpid()))
            with (EXP/f'{key}.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--run',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                save(EXP/'queue_status.json',dict(state='failed',experiment=key,exit_code=result.returncode)); return
        save(EXP/'queue_status.json',dict(state='complete'))
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
