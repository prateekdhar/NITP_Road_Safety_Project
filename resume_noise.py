import os,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))

def main():
    from ultralytics import YOLO
    from noise_training import NoiseTrainer,validate_robustness
    from experiment_200 import save
    exp=ROOT/'experiments/oct04_200'
    run=ROOT/'runs/nrdd2024_yolov9s_noise_100_oct04'
    model=YOLO(str(run/'weights/last.pt'))
    assert model.ckpt['epoch']==55, 'Unexpected checkpoint epoch; inspect before resuming'
    assert model.ckpt['train_args']['epochs']==100
    save(exp/'queue_status.json',dict(state='running',experiment='yolo26s_balanced',pid=os.getpid(),resumed_from_epoch=56))
    def progress(t):
        save(exp/'yolo26s_balanced_progress.json',dict(epoch_completed=min(t.epoch+1,t.epochs),epochs=t.epochs,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),metrics={k:float(v) for k,v in t.metrics.items()}))
    model.add_callback('on_fit_epoch_end',progress)
    try:
        model.train(trainer=NoiseTrainer,resume=True,device=0,save_dir=str(run))
        best=Path(model.trainer.best)
        metrics=YOLO(str(best)).val(data=str(exp/'baseline.yaml'),split='val',imgsz=640,batch=4,workers=0,device=0,plots=True,project=str(exp/'validation'),name='noise_resumed_640')
        save(exp/'yolo26s_balanced_complete.json',dict(best=str(best),best_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),validation={'640':{'metrics':{k:float(v) for k,v in metrics.results_dict.items()}}}))
        baseline=json.loads((exp/'yolov9s_baseline_complete.json').read_text())
        validate_robustness(ROOT,exp,{'baseline':baseline['best'],'noise_robust':str(best)})
        save(exp/'queue_status.json',dict(state='complete'))
    except Exception as e:
        save(exp/'queue_status.json',dict(state='failed',experiment='yolo26s_balanced',error=str(e)))
        raise

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
