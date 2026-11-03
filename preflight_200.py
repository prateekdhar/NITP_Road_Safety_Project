import json, os
from pathlib import Path
ROOT=Path('${PROJECT_ROOT}')
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.ultralytics')
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg

if __name__=='__main__':
    plan=json.loads((ROOT/'experiments/oct04_200/plan.json').read_text())
    weights=torch.tensor(plan['srwbce_weights'],device='cuda')
    logits=torch.randn(2,3,10,device='cuda',requires_grad=True)
    targets=torch.rand_like(logits)
    actual=torch.nn.BCEWithLogitsLoss(weight=weights,reduction='none')(logits,targets)
    expected=torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(logits[...,i],targets[...,i],reduction='none')*weights[i] for i in range(10)],-1)
    assert torch.allclose(actual,expected)
    actual.sum().backward(); assert torch.isfinite(logits.grad).all()
    for key,config in plan['configs'].items():
        source=YOLO(config['model']).model
        m=DetectionModel(source.yaml,nc=10,verbose=False).cuda().train()
        m.load(source,verbose=False); m.args=get_cfg(overrides={'epochs':100})
        criterion=m.init_criterion()
        if key=='yolo26s_balanced':
            for head in [criterion.one2many,criterion.one2one]: head.bce.weight=weights
        m.criterion=criterion
        batch={'img':torch.rand(2,3,640,640,device='cuda'),'batch_idx':torch.tensor([0,1],device='cuda'),'cls':torch.tensor([[3.],[9.]],device='cuda'),'bboxes':torch.tensor([[.5,.5,.2,.2],[.4,.4,.15,.15]],device='cuda')}
        loss,items=m(batch); assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        print(key,'CUDA forward/backward passed',items,flush=True)
        del m,source,criterion,loss,items,batch
        torch.cuda.empty_cache()
    print('PASS: weighted BCE formula, both model gradients; zero training epochs consumed.')
