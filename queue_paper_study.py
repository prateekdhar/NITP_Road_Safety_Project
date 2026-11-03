"""Run the authorized 3 x 30 epoch comparison sequentially, then validate."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from run_paper_study import save

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/'experiments/paper_noise_20261010'

def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    lock=STUDY/'active_queue.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=STUDY/('queue_'+stamp)
    out.mkdir(exist_ok=False)
    status_path=STUDY/'queue_status.json'
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'total_epochs':90,
          'variants':['augmentation','consistency','proposed'],'epochs_each':30,'batch':2,'imgsz':640,
          'output':str(out),'evaluation':'Five fixed validation conditions, unchanged reference plus all three variants; no test evaluation.'}
    save(out/'plan.json',plan)
    def execute(label,args,phase):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(status_path,{'state':phase,'current':label,'queue_pid':os.getpid(),'child_pid':proc.pid,'log':str(out/(label+'.log')),'plan':str(out/'plan.json')})
            result=proc.wait()
            if result: raise RuntimeError(f'{label} exited {result}; see {out/(label+".log")}')
    try:
        # Keep the system awake only while this authorized job is active.
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        execute('gpu_preflight',[str(ROOT/'run_paper_study.py'),'--smoke','--smoke-imgsz','640','--smoke-batch','2','--device','0','--variant','proposed','--name','gpu_preflight_'+stamp],'preflight')
        weights=['reference='+str(ROOT/'runs/nrdd2024_yolov9s_noise_100_oct04/weights/best.pt')]
        for variant in plan['variants']:
            name='paper_'+variant+'_30ep_'+stamp
            execute(variant,[str(ROOT/'run_paper_study.py'),'--variant',variant,'--epochs','30','--batch','2','--device','0','--name',name],'training')
            receipt=json.loads((STUDY/name/'status.json').read_text())
            if receipt['state']!='complete' or receipt['epochs_completed']!=30: raise RuntimeError('Incomplete variant '+variant)
            weights.append(variant+'='+receipt['best'])
        execute('paired_validation',[str(ROOT/'evaluate_paper_study.py'),'--device','0','--batch','2','--weights',*weights],'evaluating')
        save(status_path,{'state':'complete','epochs_completed':90,'plan':str(out/'plan.json'),'finished_at':datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        save(status_path,{'state':'failed','error':repr(e),'plan':str(out/'plan.json'),'finished_at':datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)

if __name__=='__main__': main()
