"""Detection adaptation of PCSD, with task-guided structure consistency.

Not a reproduction of PCSD segmentation or NER/RT-DETR. See PAPER_METHOD.md.
The teacher and all auxiliary losses are training-only; checkpoints use stock YOLO.
"""
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.metrics import bbox_iou


@dataclass
class Method:
    variant: str = 'proposed'  # augmentation, consistency, proposed
    teacher: str = ''
    curriculum_epochs: int = 10
    easy_fraction: float = .7
    kl_weight: float = .2
    feature_weight: float = .1
    structure_weight: float = .05
    min_teacher_iou: float = .3
    clean_probability: float = .3


def blur_kernel(length, angle, device, dtype):
    """Normalized discrete motion kernel; angles are multiples of 45 degrees."""
    k = torch.zeros(length, length, device=device, dtype=dtype)
    if angle == 0:
        k[length // 2, :] = 1
    elif angle == 90:
        k[:, length // 2] = 1
    elif angle == 45:
        k.fill_diagonal_(1)
    elif angle == 135:
        k = torch.eye(length, device=device, dtype=dtype).flip(1)
    else:
        raise ValueError('Supported kernel directions: 0,45,90,135')
    return k / k.sum()


def degrade_batch(images, generator=None, clean_probability=.3, condition=None):
    """Paired, label-preserving RGB stressors AFTER geometric preprocessing.

    Training: horizontal/vertical motion, brightness, heteroscedastic Gaussian
    approximation to shot/read noise, and mixtures. This is not a raw camera model.
    Fixed evaluation conditions include held-out diagonal motion and defocus.
    """
    original_dtype = images.dtype
    x = images.float().clone()
    lengths = torch.zeros(len(x), device=x.device)
    def rand():
        return float(torch.rand((), device=x.device, generator=generator))
    allowed = {None, 'clean', 'motion_diagonal', 'defocus', 'mixed_diagonal', 'sensor_noise'}
    if condition not in allowed:
        raise ValueError(condition)
    for i in range(len(x)):
        if condition == 'clean' or (condition is None and rand() < clean_probability):
            continue
        kind = condition or ('motion' if rand() < .35 else 'sensor_noise' if rand() < .5 else 'mixed')
        sample = x[i:i+1]
        if kind in ('motion', 'mixed', 'motion_diagonal', 'mixed_diagonal', 'defocus'):
            length = 9 if condition else (3, 5, 7)[min(2, int(rand()*3))]
            angle = 45 if condition else (0 if rand() < .5 else 90)
            kernel = blur_kernel(length, angle, x.device, x.dtype)
            if kind == 'defocus':
                v = torch.arange(length, device=x.device) - length//2
                kernel = ((v[:, None]**2 + v[None, :]**2) <= (length//2)**2).float()
                kernel /= kernel.sum()
            sample = F.conv2d(F.pad(sample, (length//2,)*4, mode='reflect'), kernel.expand(3, 1, length, length), groups=3)
            lengths[i] = length
        if kind in ('mixed', 'mixed_diagonal', 'sensor_noise'):
            brightness = .4 if condition else .4 + .6*rand()
            sample = sample * brightness
            read_sigma = .025 if condition else .005 + .025*rand()
            shot_scale = .02 if condition else .003 + .017*rand()
            std = (read_sigma**2 + shot_scale*sample.clamp_min(0)).sqrt()
            sample = sample + torch.randn(sample.shape, device=x.device, generator=generator) * std
        x[i:i+1] = sample.clamp(0, 1)
    return x.to(original_dtype), lengths


def bernoulli_kl(student_logits, teacher_logits):
    """Independent sigmoid classes, not a softmax over mutually exclusive anchors."""
    q = teacher_logits.detach().float().sigmoid().clamp(1e-6, 1-1e-6)
    s = student_logits.float()
    return (q*(q.log()-F.logsigmoid(s)) + (1-q)*((1-q).log()-F.logsigmoid(-s))).mean(-1)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class GuidedLoss:
    """Wrap the installed detection criterion without modifying site-packages."""
    def __init__(self, model, method):
        self.base = v8DetectionLoss(model)
        self.method = method
        self.epoch = 0

    def __call__(self, predictions, batch):
        losses, items = self.base(predictions, batch)
        student = self.base.parse_output(predictions)
        zero = student['scores'].float().sum() * 0
        kl = feature = structure = zero
        if 'teacher_preds' in batch:
            teacher = batch['teacher_preds']
            weights = self.target_weights(teacher, batch)
            kl = weighted_mean(bernoulli_kl(student['scores'].transpose(1, 2), teacher['scores'].transpose(1, 2)), weights)
            offset = 0
            feat_values = []; structure_values = []
            for s, t in zip(student['feats'], teacher['feats']):
                b, c, h, w = s.shape
                local_weights = weights[:, offset:offset+h*w].reshape(b, h, w)
                offset += h*w
                sn = F.normalize(s.float(), dim=1, eps=1e-6)
                tn = F.normalize(t.detach().float(), dim=1, eps=1e-6)
                feat_values.append(weighted_mean((sn-tn).square().sum(1), local_weights))
                if self.method.variant == 'proposed':
                    # Two-axis local feature differences; these are NOT crack masks.
                    dx = ((sn[:,:,:,1:]-sn[:,:,:,:-1])-(tn[:,:,:,1:]-tn[:,:,:,:-1])).square().sum(1)
                    dy = ((sn[:,:,1:,:]-sn[:,:,:-1,:])-(tn[:,:,1:,:]-tn[:,:,:-1,:])).square().sum(1)
                    wx = torch.minimum(local_weights[:,:,1:], local_weights[:,:,:-1])
                    wy = torch.minimum(local_weights[:,1:,:], local_weights[:,:-1,:])
                    structure_values.append((weighted_mean(dx,wx)+weighted_mean(dy,wy))*.5)
            feature = torch.stack(feat_values).mean()
            if structure_values: structure = torch.stack(structure_values).mean()
        ramp = min(1., (self.epoch+1)/max(1,self.method.curriculum_epochs))
        aux = ramp*(self.method.kl_weight*kl + self.method.feature_weight*feature + self.method.structure_weight*structure)
        # Stock criterion returns three components scaled by batch size.
        losses = torch.cat((losses.reshape(-1), (aux*len(batch['img'])).reshape(1)))
        items = dict(items, guide_loss=aux.detach())
        return losses, items

    @torch.no_grad()
    @torch.amp.autocast('cuda', enabled=False)
    def target_weights(self, teacher, batch):
        # Matching uses fp32 throughout: autocast would otherwise turn the DFL
        # projection matmul back into fp16, conflicting with fp32 IoU buffers.
        scores = teacher['scores'].float().permute(0,2,1)
        anchors, strides = make_anchors(teacher['feats'], self.base.stride, .5)
        h,w = batch['img'].shape[-2:]
        raw = torch.cat((batch['batch_idx'].view(-1,1),batch['cls'].view(-1,1),batch['bboxes']),1)
        gt = self.base.preprocess(raw, len(scores), torch.tensor([w,h,w,h], device=scores.device))
        if not gt.shape[1]: return torch.zeros(scores.shape[:2],device=scores.device)
        labels, boxes = gt.split((1,4),2)
        pred_boxes = self.base.bbox_decode(anchors,teacher['boxes'].float().permute(0,2,1))*strides
        assigned_labels, assigned_boxes, _, fg, indices = self.base.assigner(scores.sigmoid(), pred_boxes, anchors*strides, labels, boxes, boxes.sum(2,keepdim=True)>0)
        assigned_labels = assigned_labels.long().clamp(0,self.base.nc-1)
        confidence = scores.sigmoid().gather(2,assigned_labels.unsqueeze(-1)).squeeze(-1)
        iou = bbox_iou(pred_boxes,assigned_boxes,xywh=False).squeeze(-1)
        reliable = fg & (scores.argmax(2)==assigned_labels) & (iou>=self.method.min_teacher_iou)
        if self.method.variant == 'proposed' and reliable.any():
            progress = min(1., self.epoch/max(1,self.method.curriculum_epochs))
            proportion = self.method.easy_fraction+(1-self.method.easy_fraction)*progress
            # Detection adaptation: per-batch reliable-positive quantile, not paper's
            # full-training-set pixel threshold. GT loss is never masked.
            difficulty = 1-confidence
            threshold = torch.quantile(difficulty[reliable], proportion)
            reliable &= difficulty<=threshold
        weights = reliable.float()
        if self.method.variant == 'proposed':
            size = (assigned_boxes[...,2:]-assigned_boxes[...,:2]).amin(-1).clamp_min(1)
            weights *= 1+(batch['blur_extent'][:,None]/size).clamp(0,2)
        # Normalize by anchors per assigned GT object so large objects cannot dominate.
        obj_ids = indices+torch.arange(len(scores),device=scores.device)[:,None]*gt.shape[1]
        counts = torch.bincount(obj_ids[reliable],minlength=len(scores)*gt.shape[1]).clamp_min(1)
        weights /= counts[obj_ids]
        return weights


class PaperTrainer(DetectionTrainer):
    def __init__(self, *args, method_config=None, **kwargs):
        self.method = Method(**(method_config or {}))
        if self.method.variant not in ('augmentation','consistency','proposed'):
            raise ValueError(self.method.variant)
        super().__init__(*args, **kwargs)
        self.add_callback('on_train_start', self.install_guidance)
        self.add_callback('on_train_epoch_start', self.set_guidance_epoch)

    def install_guidance(self, trainer):
        if self.method.variant == 'augmentation': return
        from ultralytics import YOLO
        self.teacher = YOLO(self.method.teacher).model.to(self.device).float().eval()
        self.teacher.requires_grad_(False)
        self.model.criterion = GuidedLoss(self.model,self.method)
        # Isolated generator keeps the augmentation sequence matched across variants.
        self.corruption_generator = torch.Generator(device=self.device).manual_seed(self.args.seed+10000)

    def set_guidance_epoch(self, trainer):
        if isinstance(getattr(self.model,'criterion',None), GuidedLoss):
            self.model.criterion.epoch = self.epoch
        # Deterministic per-epoch RNG also supports resumed runs.
        self.corruption_generator = torch.Generator(device=self.device).manual_seed(self.args.seed+10000+self.epoch)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        clean = batch['img']
        if self.method.variant != 'augmentation':
            with torch.no_grad():
                result = self.teacher(clean)
                batch['teacher_preds'] = result[1] if isinstance(result,tuple) else result
        batch['img'], batch['blur_extent'] = degrade_batch(clean,self.corruption_generator,self.method.clean_probability)
        return batch
