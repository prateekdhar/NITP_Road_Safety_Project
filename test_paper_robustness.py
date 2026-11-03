"""Numerical and gradient checks, including teacher/GT disagreement and empty GT."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
import unittest
import torch
from types import SimpleNamespace
from paper_robustness import degrade_batch, bernoulli_kl, weighted_mean, blur_kernel, GuidedLoss, Method

class RobustnessChecks(unittest.TestCase):
    def test_paired_corruption_deterministic_and_bounded(self):
        x=torch.rand(3,3,32,32); before=x.clone()
        a,l=degrade_batch(x,torch.Generator().manual_seed(41),condition='mixed_diagonal')
        b,m=degrade_batch(x,torch.Generator().manual_seed(41),condition='mixed_diagonal')
        self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.equal(l,m))
        self.assertTrue(torch.equal(x,before)); self.assertEqual(a.shape,x.shape)
        self.assertTrue(torch.isfinite(a).all()); self.assertTrue((a>=0).all() and (a<=1).all())
        self.assertTrue((l==9).all())

    def test_identity_and_kernel(self):
        x=torch.rand(2,3,32,32)
        y,l=degrade_batch(x,condition='clean')
        self.assertTrue(torch.equal(x,y)); self.assertTrue((l==0).all())
        for angle in (0,45,90,135):
            self.assertAlmostEqual(float(blur_kernel(7,angle,'cpu',torch.float32).sum()),1,places=6)

    def test_kl_teacher_detached_and_empty_mask(self):
        s=torch.randn(2,5,10,requires_grad=True); t=torch.randn(2,5,10,requires_grad=True)
        kl=bernoulli_kl(s,t)
        self.assertTrue((kl>=-1e-6).all())
        kl.mean().backward(); self.assertIsNotNone(s.grad); self.assertIsNone(t.grad)
        self.assertAlmostEqual(float(bernoulli_kl(t,t).mean().detach()),0,places=6)
        zero=weighted_mean(kl.detach(),torch.zeros(2,5))
        self.assertEqual(float(zero),0)

    def test_teacher_disagreement_is_not_distilled(self):
        guide=GuidedLoss.__new__(GuidedLoss); guide.method=Method(variant='consistency'); guide.epoch=0
        gt=torch.tensor([[[0.,0.,0.,16.,16.]]])
        box=torch.tensor([[[0.,0.,16.,16.]]]).expand(1,4,4)
        guide.base=SimpleNamespace(nc=10,stride=torch.tensor([8.]),
            preprocess=lambda *args:gt,
            bbox_decode=lambda *args:box/8,
            assigner=lambda *args:(torch.zeros(1,4,dtype=torch.long),box,None,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4,dtype=torch.long)))
        scores=torch.full((1,10,4),-3.); scores[:,0,:]=3.
        teacher={'scores':scores,'boxes':torch.zeros(1,64,4),'feats':[torch.zeros(1,4,2,2)]}
        batch={'img':torch.zeros(1,3,16,16),'batch_idx':torch.tensor([0]),'cls':torch.tensor([[0]]),'bboxes':torch.tensor([[.5,.5,1.,1.]]),'blur_extent':torch.tensor([3.])}
        self.assertGreater(float(guide.target_weights(teacher,batch).sum()),0)
        teacher['scores'][:,1,:]=6. # Confident wrong class must never be trusted.
        self.assertEqual(float(guide.target_weights(teacher,batch).sum()),0)
        guide.base.preprocess=lambda *args:torch.zeros(1,0,5)
        self.assertEqual(float(guide.target_weights(teacher,batch).sum()),0)

    def test_ground_truth_loss_survives_missing_teacher(self):
        class Base:
            def __call__(self,*args):return torch.tensor([1.,2.,3.]),{'box_loss':torch.tensor(1.),'cls_loss':torch.tensor(2.),'dfl_loss':torch.tensor(3.)}
            def parse_output(self,p):return p
        guide=GuidedLoss.__new__(GuidedLoss); guide.base=Base(); guide.method=Method(); guide.epoch=0
        losses,items=guide({'scores':torch.zeros(1,10,4,requires_grad=True)}, {'img':torch.zeros(1,3,16,16)})
        self.assertEqual(float(losses.sum().detach()),6.)
        self.assertEqual(float(items['guide_loss']),0.)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
