"""Local, checkpoint-serializable YOLO26 ablation components; no package edits."""
import os,re,copy,json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import yaml
import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Concat

class WeightedConcat(nn.Module):
    """Learned positive branch weights, normalized to mean one, then concatenate.

    This is branch-weighted concatenation, NOT a full BiFPN implementation.
    Equal initial logits exactly recover ordinary concatenation.
    """
    def __init__(self,n=2,dimension=1):
        super().__init__(); self.logits=nn.Parameter(torch.zeros(n)); self.d=dimension
    def forward(self,x):
        w=self.logits.softmax(0)*len(x)
        return torch.cat([v*w[i].to(v.dtype) for i,v in enumerate(x)],self.d)

def p2_config():
    p=Path(ultralytics.__file__).parent/'cfg/models/26/yolo26.yaml'
    d=yaml.safe_load(p.read_text(encoding='utf-8'))
    d['scale']='s'
    # Preserve all existing backbone/neck indices. Add a separate top-down P2 branch.
    d['head']=d['head'][:-1]+[
        [16,1,'nn.Upsample',[None,2,'nearest']],
        [[-1,2],1,'Concat',[1]],
        [-1,2,'C3k2',[128,True]],
        [[25,16,19,22],1,'Detect',['nc']]]
    return d

class AblationTrainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        self.variant=json.loads(os.environ.get('RDD_ABLATION_SPEC','{}'))
        p2=self.variant.get('p2',False)
        model=super().get_model(p2_config() if p2 else cfg,weights=None,verbose=verbose)
        if weights is not None:
            source=weights.float().state_dict(); target=model.state_dict(); transfer={}
            for k,v in source.items():
                dest=k
                if p2 and k.startswith('model.23.'):
                    rest=k[len('model.23.'):]
                    rest=re.sub(r'^((?:one2one_)?cv[23])\.(\d+)\.',lambda m:f'{m[1]}.{int(m[2])+1}.',rest)
                    dest='model.26.'+rest
                if dest in target and target[dest].shape==v.shape: transfer[dest]=v
            model.load_state_dict(transfer,strict=False)
            model.ablation_transfer={'matched_tensors':len(transfer),'target_tensors':len(target),'unmatched_keys':[k for k in target if k not in transfer]}
        if self.variant.get('fusion'):
            for i,m in enumerate(model.model):
                if isinstance(m,Concat):
                    replacement=WeightedConcat(len(m.f),m.d)
                    for attr in ['i','f','type','np']:
                        if hasattr(m,attr): setattr(replacement,attr,getattr(m,attr))
                    replacement.type='ablation_components.WeightedConcat'
                    model.model[i]=replacement
        model.ablation_variant=self.variant
        return model
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        self.args.workers=0
        return super().get_dataloader(dataset_path,batch_size,rank,mode)
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        prob=self.variant.get('noise_probability',0)
        if prob:
            x=batch['img'].clone()
            for i in range(len(x)):
                if torch.rand((),device=x.device)<prob:
                    if torch.rand((),device=x.device)<.5:
                        sigma=torch.empty((),device=x.device).uniform_(.005,.03)
                        x[i]=(x[i]+torch.randn_like(x[i])*sigma).clamp(0,1)
                    else:
                        k=x.new_tensor([1.,2.,1.]); kernel=(k[:,None]*k[None,:]/16).expand(3,1,3,3)
                        x[i:i+1]=F.conv2d(F.pad(x[i:i+1],(1,1,1,1),mode='reflect'),kernel,groups=3)
            batch['img']=x
        return batch
