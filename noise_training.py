"""Photometric robustness experiment; geometry and labels are unchanged."""
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

def degrade(images, kind, severity, generator=None):
    x=images.float()
    if kind=='sensor_noise':
        noise=torch.randn(x.shape,device=x.device,generator=generator)
        return (x+noise*severity).clamp(0,1).to(images.dtype)
    if kind=='lowlight_blur':
        # Fixed Gaussian blur and reduced contrast/brightness; preserve spatial coordinates.
        v=torch.tensor([1.,2.,1.],device=x.device)
        kernel=(v[:,None]*v[None,:]/16).expand(3,1,3,3)
        x=F.conv2d(F.pad(x,(1,1,1,1),mode='reflect'),kernel,groups=3)
        x=(x-.5)*.8+.5
        return (x*severity).clamp(0,1).to(images.dtype)
    raise ValueError(kind)

class NoiseTrainer(DetectionTrainer):
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        images=batch['img'].clone()
        # Independently leave half the images unchanged. Others receive one corruption.
        choices=torch.rand(images.shape[0],device=images.device)
        for i in range(images.shape[0]):
            if choices[i]<.25:
                sigma=float(torch.empty((),device=images.device).uniform_(.01,.06))
                images[i:i+1]=degrade(images[i:i+1],'sensor_noise',sigma)
            elif choices[i]<.5:
                light=float(torch.empty((),device=images.device).uniform_(.35,.75))
                images[i:i+1]=degrade(images[i:i+1],'lowlight_blur',light)
        batch['img']=images
        return batch

class NoiseValidator(DetectionValidator):
    kind='sensor_noise'
    severity=.03
    def preprocess(self,batch):
        batch=super().preprocess(batch)
        if not hasattr(self,'noise_generator'):
            self.noise_generator=torch.Generator(device=batch['img'].device).manual_seed(20261004)
        batch['img']=degrade(batch['img'],self.kind,self.severity,self.noise_generator)
        return batch

def validate_robustness(root,exp,weights):
    import json
    from ultralytics import YOLO
    report={}
    for model_name,path in weights.items():
        report[model_name]={}
        for kind,value in [('clean',0),('sensor_noise',.03),('sensor_noise',.06),('lowlight_blur',.6),('lowlight_blur',.35)]:
            kwargs={}
            if kind!='clean':
                cls=type('FixedNoiseValidator',(NoiseValidator,),{'kind':kind,'severity':value})
                kwargs['validator']=cls
            result=YOLO(str(path)).val(data=str(exp/'baseline.yaml'),split='val',imgsz=640,batch=4,workers=0,device=0,plots=False,project=str(exp/'robustness'),name=f'{model_name}_{kind}_{value}',**kwargs)
            report[model_name][f'{kind}_{value}']={k:float(v) for k,v in result.results_dict.items()}
            (exp/'robustness_comparison.json').write_text(json.dumps(report,indent=2))
    return report
