"""Validation-only paired unseen-corruption evaluation; never uses test split."""
import argparse, json, os
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',nargs='+',required=True,help='label=absolute_checkpoint_path for each model')
    p.add_argument('--device',default='0')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--smoke',action='store_true',help='Use the newest isolated smoke dataset, never research results')
    p.add_argument('--resume-dir',type=Path,help='Resume saved conditions in an interrupted evaluation')
    a=p.parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from paper_robustness import degrade_batch
    from run_paper_study import save,digest
    study=ROOT/'experiments/paper_noise_20261010'
    out=a.resume_dir if a.resume_dir else study/(('smoke_evaluation_' if a.smoke else 'evaluation_')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not a.resume_dir: out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'experiments/oct04_200/baseline.yaml'
    if a.smoke:
        candidates=list(study.glob('*/smoke.yaml'))
        if not candidates: raise RuntimeError('Run training smoke test first')
        data=max(candidates,key=lambda p:p.stat().st_mtime)
    size=160 if a.smoke else 640
    torch.set_num_threads(2)
    results={'split':'val','smoke_only':a.smoke,'data':str(data),'imgsz':size,'batch':a.batch,'seed':20261010,'models':{},'note':'Post-resize synthetic RGB stressors, not real-camera evidence. Fixed diagonal motion/defocus are held out from this training recipe.'}
    if a.resume_dir:
        import json
        saved=json.loads((out/'comparison.json').read_text())
        for key in ['split','smoke_only','data','imgsz','batch','seed']: assert saved[key]==results[key],key
        assert set(saved['models'])=={item.split('=',1)[0] for item in a.weights}
        results=saved
    class StressValidator(DetectionValidator):
        condition='clean'
        def preprocess(self,batch):
            batch=super().preprocess(batch)
            if not hasattr(self,'stress_rng'):
                self.stress_rng=torch.Generator(device=batch['img'].device).manual_seed(20261010)
            batch['img'],_=degrade_batch(batch['img'],self.stress_rng,condition=self.condition)
            return batch
    for item in a.weights:
        label,weights=item.split('=',1)
        if not label.replace('_','').isalnum() or (label in results['models'] and not a.resume_dir): raise ValueError('Unique simple labels required')
        model_results=results['models'].get(label,{'weights':weights,'sha256':digest(weights),'conditions':{}})
        assert model_results['sha256']==digest(weights)
        results['models'][label]=model_results
        for condition in ['clean','sensor_noise','motion_diagonal','defocus','mixed_diagonal']:
            if condition in model_results['conditions']:
                print('Already saved:',label,condition,flush=True)
                continue
            validator=type('FixedStressValidator',(StressValidator,),{'condition':condition})
            m=YOLO(weights).val(validator=validator,data=str(data),split='val',imgsz=size,
                                batch=a.batch,device=a.device,workers=0,plots=False,save_json=True,project=str(out),name=label+'_'+condition)
            model_results['conditions'][condition]={'metrics':{k:float(v) for k,v in m.results_dict.items()},'speed':m.speed,
                'per_class':[{'class_id':int(c),'precision':float(m.box.p[i]),'recall':float(m.box.r[i]),'ap50':float(m.box.ap50[i]),'ap50_95':float(m.box.ap[i])} for i,c in enumerate(m.box.ap_class_index)]}
            save(out/'comparison.json',results)
    save(out/'status.json',{'state':'complete'})
    print(out)

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
