"""Resume the user-paused 90-epoch study without extending its budget."""
import os, sys, json, subprocess, functools, ctypes, shutil
from pathlib import Path
from datetime import datetime, timezone
from run_paper_study import save, digest

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/'experiments/paper_noise_20261010'
RUN=STUDY/'paper_consistency_30ep_20261010_184826'
CHECKPOINT=RUN/'pause-20261011-173427/resume.pt'
EXPECTED='7b21d1ee82c9ea82a92cfafd27c83aa1f92892fe0538b845099c32f3aec356df'

def worker():
    import torch
    from ultralytics import YOLO
    from paper_robustness import PaperTrainer
    torch.set_num_threads(2)
    assert digest(CHECKPOINT)==EXPECTED
    config=json.loads((RUN/'configuration.json').read_text())
    assert config['method']['variant']=='consistency'
    model=YOLO(str(CHECKPOINT))
    assert model.ckpt['epoch']==24 and model.ckpt.get('optimizer') is not None
    assert model.ckpt['train_args']['epochs']==30
    def progress(t):
        save(RUN/'status.json',{'state':'training','epoch_completed':min(t.epoch+1,t.epochs),'epochs':30,'pid':os.getpid(),
             'resumed_from_epoch':25,'updated_at':datetime.now(timezone.utc).isoformat(),'metrics':{k:float(v) for k,v in t.metrics.items()}})
    def verify(t):
        assert t.start_epoch==25 and t.epochs==30
        assert Path(t.save_dir).resolve()==(RUN/'train').resolve()
        save(RUN/'resume_verified.json',{'start_epoch_one_based':26,'target_epochs':30,'optimizer':type(t.optimizer).__name__,
             'save_dir':str(t.save_dir),'checkpoint_sha256':EXPECTED,'pid':os.getpid()})
    model.add_callback('on_fit_epoch_end',progress)
    model.add_callback('on_train_start',verify)
    model.train(trainer=functools.partial(PaperTrainer,method_config=config['method']),resume=True,device=0,save_dir=str(RUN/'train'))
    best=Path(model.trainer.best)
    loaded=YOLO(str(best)); assert type(loaded.model).__module__=='ultralytics.nn.tasks'
    assert not hasattr(loaded.model,'teacher')
    save(RUN/'status.json',{'state':'complete','smoke_only':False,'epochs_completed':30,'best':str(best),'sha256':digest(best),
         'resumed_from_epoch':25,'finished_at':datetime.now(timezone.utc).isoformat()})

def main():
    if '--worker' in sys.argv: return worker()
    assert digest(CHECKPOINT)==EXPECTED
    with (STUDY/'active_queue.lock').open('x') as f:f.write(str(os.getpid()))
    out=STUDY/('resumed_queue_'+datetime.now().strftime('%Y%m%d_%H%M%S')); out.mkdir()
    save(out/'plan.json',{'total_epoch_budget':90,'completed_before_resume':55,'remaining_training_epochs':35,
         'augmentation_target':30,'consistency_target':30,'proposed_target':30,'resume_checkpoint':str(CHECKPOINT)})
    shutil.copy2(RUN/'train/results.csv',out/'consistency_results_before_resume.csv')
    def execute(label,args):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(STUDY/'queue_status.json',{'state':'evaluating' if label=='paired_validation' else 'training','current':label,
                 'queue_pid':os.getpid(),'child_pid':child.pid,'log':str(out/(label+'.log')),'plan':str(out/'plan.json')})
            if child.wait()!=0:raise RuntimeError(label+' failed; inspect '+str(out/(label+'.log')))
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        execute('consistency',[str(Path(__file__)),'--worker'])
        r=json.loads((RUN/'status.json').read_text());assert r['state']=='complete'
        weights=['reference='+str(ROOT/'runs/nrdd2024_yolov9s_noise_100_oct04/weights/best.pt'),'augmentation='+json.loads((STUDY/'paper_augmentation_30ep_20261010_184826/status.json').read_text())['best'],'consistency='+r['best']]
        for variant in ['proposed']:
            name=f'paper_{variant}_30ep_20261010_184826'
            execute(variant,[str(ROOT/'run_paper_study.py'),'--variant',variant,'--epochs','30','--batch','2','--device','0','--name',name])
            r=json.loads((STUDY/name/'status.json').read_text());assert r['state']=='complete' and r['epochs_completed']==30
            weights.append(variant+'='+r['best'])
        execute('paired_validation',[str(ROOT/'evaluate_paper_study.py'),'--device','0','--batch','2','--weights',*weights])
        save(STUDY/'queue_status.json',{'state':'complete','epochs_completed':90,'plan':str(out/'plan.json')})
    except Exception as e:
        save(STUDY/'queue_status.json',{'state':'failed','error':repr(e),'plan':str(out/'plan.json')});raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        (STUDY/'active_queue.lock').unlink(missing_ok=True)

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
