"""Sequential full-budget validation-only YOLO26s study."""
import os,json,csv,sys,hashlib,subprocess,ctypes,argparse
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/yolo26s_ablation_20261021'
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
os.environ.setdefault('WANDB_MODE','disabled')
BASE=dict(data=str(ROOT/'experiments/oct04_200/baseline.yaml'),epochs=100,patience=0,imgsz=640,batch=4,device=0,workers=0,amp=True,cache=False,seed=42,deterministic=True,optimizer='MuSGD',lr0=.01,momentum=.9,weight_decay=.0005,warmup_bias_lr=0.0,mosaic=1.0,close_mosaic=10,save=True,save_period=10,plots=True,exist_ok=False)
SPECS={
 'control':{},'lr_half':{'lr0':.005},'lr_double':{'lr0':.02},
 'wd_low':{'weight_decay':.0001},'wd_high':{'weight_decay':.001},
 'mosaic_025':{'mosaic':.25},'mosaic_050':{'mosaic':.5},
 'close_mosaic_20':{'close_mosaic':20},'resolution_800':{'imgsz':800},
 'noise_010':{'noise_probability':.10},'noise_025':{'noise_probability':.25},
 'p2':{'p2':True},'fusion':{'fusion':True},'p2_fusion':{'p2':True,'fusion':True}}
ARCH=['p2','fusion','p2_fusion']
ELIGIBLE=['control','lr_half','lr_double','wd_low','wd_high','mosaic_025','mosaic_050','close_mosaic_20']
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8');t.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def history(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def config(key,smoke=False):
 spec=dict(SPECS[key]);recipe={}
 if key in ARCH and not smoke:
  chosen=json.loads((EXP/'selected_recipe.json').read_text());recipe=SPECS[chosen['key']]
 cfg=dict(BASE,**{})
 cfg.update({k:v for k,v in recipe.items() if k in BASE});cfg.update({k:v for k,v in spec.items() if k in BASE})
 cfg.update(project=str(EXP/('preflight' if smoke else 'runs')),name=key)
 if smoke:cfg.update(data=str(ROOT/'experiments/paper_noise_20261010/gpu_preflight_20261010_184826/smoke.yaml'),epochs=1,close_mosaic=0,plots=False,save_period=-1)
 return cfg,spec
def run(key,smoke=False,resume_checkpoint=None):
 import torch,ultralytics
 from ultralytics import YOLO
 from ultralytics.utils.torch_utils import get_flops
 from ablation_components import AblationTrainer,WeightedConcat
 assert ultralytics.__version__=='8.4.161'
 assert torch.cuda.is_available()
 torch.set_num_threads(2)
 assert digest(ROOT/'data/processed/split_manifest.json')=='25a8ab1f25b485b3db90a6bcfd1d9ae6c44ccaa53479e8dc3561780f7b68817e'
 cfg,spec=config(key,smoke);runpath=Path(cfg['project'])/key
 if not resume_checkpoint: assert not runpath.exists(),f'Refusing overwrite: {runpath}'
 os.environ['RDD_ABLATION_SPEC']=json.dumps(spec)
 source=ROOT/'yolo26s.pt';model=YOLO(str(resume_checkpoint or source))
 start_epoch=0
 if resume_checkpoint:
  assert not smoke and key not in ARCH, 'Architectural resume requires separate validation'
  assert Path(resume_checkpoint).resolve()==(runpath/'weights/last.pt').resolve()
  assert model.ckpt.get('optimizer') is not None, 'Optimizer state missing'
  start_epoch=int(model.ckpt['epoch'])+1
  assert 0<start_epoch<100 and model.ckpt['train_args']['epochs']==100
  prior=history(runpath/'results.csv')
  assert len(prior)==start_epoch and int(prior[-1]['epoch'])==start_epoch
  previous=json.loads((runpath/'provenance.json').read_text())
  assert previous['spec']==spec
  for setting in ['imgsz','batch','optimizer','lr0','weight_decay','mosaic','close_mosaic','seed']:
   assert model.ckpt['train_args'][setting]==cfg[setting],setting
 receipt=EXP/('preflight_status' if smoke else 'status')/(key+'.json')
 meta={'key':key,'source':str(source),'source_sha256':digest(source),'spec':spec,'config':cfg,'started_at':now(),'smoke_only':smoke}
 if resume_checkpoint: meta.update(resumed_from_epoch=start_epoch,resume_checkpoint=str(resume_checkpoint),resume_sha256=digest(resume_checkpoint),original_started_at=previous['started_at'],resume_note='Optimizer and checkpoint restored; RNG trajectory may differ after restart')
 def started(t):
  assert t.epochs==cfg['epochs'] and t.data['nc']==10 and t.start_epoch==start_epoch
  assert Path(t.save_dir).resolve()==runpath.resolve()
  strides=t.model.stride.tolist();assert strides==([4.,8.,16.,32.] if spec.get('p2') else [8.,16.,32.]),strides
  fusion=[m for m in t.model.modules() if isinstance(m,WeightedConcat)]
  assert bool(fusion)==bool(spec.get('fusion'))
  meta.update(parameters=sum(p.numel() for p in t.model.parameters()),gflops=get_flops(t.model,cfg['imgsz']),strides=strides,transfer=t.model.ablation_transfer,optimizer=type(t.optimizer).__name__)
  assert t.optimizer.param_groups[0]['initial_lr']>0
  save(runpath/('resume_provenance.json' if resume_checkpoint else 'provenance.json'),meta)
 def progress(t):
  r=history(runpath/'results.csv');save(receipt,dict(meta,state='training',epochs_completed=int(r[-1]['epoch']),updated_at=now(),metrics=t.metrics))
 model.add_callback('on_train_start',started);model.add_callback('on_fit_epoch_end',progress)
 if resume_checkpoint:
  model.train(trainer=AblationTrainer,resume=True,device=0,workers=0,save_dir=str(runpath),cache=False)
 else:
  model.train(trainer=AblationTrainer,**cfg)
 rows=history(runpath/'results.csv');assert len(rows)==cfg['epochs'] and rows[-1]['epoch']==cfg['epochs']
 best=max(rows,key=lambda r:r['metrics/mAP50-95(B)']);ckpt=runpath/'weights/best.pt'
 # A fresh model object must deserialize the custom layer and run prediction.
 restored=YOLO(str(ckpt));restored.predict(source=torch.zeros(1,3,128,128),imgsz=128,device=0,verbose=False)
 report=dict(meta,state='complete',epochs_completed=len(rows),best_epoch=best,latest_epoch=rows[-1],best=str(ckpt),sha256=digest(ckpt),finished_at=now(),checkpoint_reload_passed=True)
 if not smoke:
  result=restored.val(data=BASE['data'],split='val',imgsz=cfg['imgsz'],batch=4,device=0,workers=0,plots=True,project=str(EXP/'validation'),name=key)
  report['validation']={'metrics':result.results_dict,'speed':result.speed,'per_class':[dict(class_id=int(c),precision=float(result.box.p[i]),recall=float(result.box.r[i]),ap50=float(result.box.ap50[i]),ap50_95=float(result.box.ap[i])) for i,c in enumerate(result.box.ap_class_index)]}
 save(receipt,report)
def queue():
 EXP.mkdir(parents=True,exist_ok=True)
 with (EXP/'active_queue.lock').open('x') as f:f.write(str(os.getpid()))
 try:
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
  def execute(key,smoke=False):
   receipt=EXP/('preflight_status' if smoke else 'status')/(key+'.json')
   if receipt.exists() and json.loads(receipt.read_text()).get('state')=='complete':return
   log=EXP/'logs'/f"{key}_{'preflight' if smoke else 'training'}.log";log.parent.mkdir(exist_ok=True)
   with log.open('a',encoding='utf-8') as out:
    args=[sys.executable,'-u',str(Path(__file__)),'--run',key]+(['--smoke'] if smoke else [])
    requests=EXP/'resume_requests.json'
    if not smoke and requests.exists():
     checkpoint=json.loads(requests.read_text()).get(key)
     if checkpoint:args.extend(['--resume-checkpoint',checkpoint])
    child=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    save(EXP/'queue_status.json',dict(state='preflight' if smoke else 'training',current=key,pid=os.getpid(),child_pid=child.pid,log=str(log),updated_at=now()))
    code=child.wait()
   if code:raise RuntimeError(f'{key} failed ({code}); see {log}')
  for key in ['control','resolution_800','noise_025','p2','fusion','p2_fusion']:execute(key,True)
  for key in SPECS:
   if key in ARCH and not (EXP/'selected_recipe.json').exists():
    candidates={k:json.loads((EXP/'status'/f'{k}.json').read_text())['best_epoch']['metrics/mAP50-95(B)'] for k in ELIGIBLE}
    chosen=max(candidates,key=candidates.get)
    save(EXP/'selected_recipe.json',dict(key=chosen,criterion='highest clean validation mAP50-95; ties use prespecified order',candidates=candidates,overrides=SPECS[chosen]))
   execute(key)
  save(EXP/'queue_status.json',dict(state='complete',finished_at=now(),runs=len(SPECS),epochs=100*len(SPECS)))
 except Exception as e:
  save(EXP/'failure.json',dict(error=str(e),time=now()))
  save(EXP/'queue_status.json',dict(state='failed',error=str(e),updated_at=now()))
  raise
 finally:
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);(EXP/'active_queue.lock').unlink(missing_ok=True)
if __name__=='__main__':
 import multiprocessing
 multiprocessing.freeze_support()
 p=argparse.ArgumentParser();p.add_argument('--queue',action='store_true');p.add_argument('--run',choices=SPECS);p.add_argument('--smoke',action='store_true');p.add_argument('--resume-checkpoint');a=p.parse_args()
 if a.queue:queue()
 elif a.run:run(a.run,a.smoke,a.resume_checkpoint)
 else:p.error('Choose --queue or --run')
