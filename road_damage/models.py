"""Paper-inspired experimental modules; not an exact reproduction of YOLOv8-PD."""
import math

import torch
from torch import nn
from ultralytics.nn.modules import C2f, Detect, GhostBottleneck
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


class BOT(nn.Module):
    """Residual global attention on the final, low-resolution backbone map."""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = (x + self.position(x)).flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + attended.transpose(1, 2).reshape(b, c, h, w)


class LSKA(nn.Module):
    """Separable large-kernel spatial attention with a residual gate."""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, (1,5), padding=(0,2), groups=channels),
            nn.Conv2d(channels, channels, (5,1), padding=(2,0), groups=channels),
            nn.Conv2d(channels, channels, (1,7), padding=(0,9), dilation=3, groups=channels),
            nn.Conv2d(channels, channels, (7,1), padding=(9,0), dilation=3, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x + x * self.attention(x)


class C2fGhost(C2f):
    def __init__(self, c1, c2, n=1):
        super().__init__(c1, c2, n)
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))


def group_conv(c1, c2, kernel=3):
    return nn.Sequential(nn.Conv2d(c1, c2, kernel, padding=kernel//2, bias=False),
                         nn.GroupNorm(8, c2), nn.SiLU())


class LSCDHead(Detect):
    """Shared GroupNorm convolutions with per-level regression scales."""
    def __init__(self, nc=4, ch=(), hidden=64):
        super().__init__(nc, ch)
        del self.cv2
        del self.cv3
        self.projections = nn.ModuleList(group_conv(c, hidden, 1) for c in ch)
        self.shared = nn.Sequential(group_conv(hidden, hidden), group_conv(hidden, hidden))
        self.box = nn.Conv2d(hidden, 4*self.reg_max, 1)
        self.classifier = nn.Conv2d(hidden, nc, 1)
        self.scales = nn.Parameter(torch.ones(self.nl))

    def bias_init(self):
        nn.init.constant_(self.box.bias, 1.0)
        nn.init.constant_(self.classifier.bias, math.log(5 / self.nc / (640/16)**2))

    def forward(self, x):
        outputs = []
        for i, feature in enumerate(x):
            shared = self.shared(self.projections[i](feature))
            outputs.append(torch.cat((self.box(shared) * self.scales[i], self.classifier(shared)), 1))
        if self.training:
            return outputs
        decoded = self._inference(outputs)
        return decoded if self.export else (decoded, outputs)


VARIANTS = {"baseline", "bot", "lska", "ghost", "lscd", "pd"}


def replace_layer(model, index, replacement):
    original = model.model[index]
    for key in ("i", "f"):
        setattr(replacement, key, getattr(original, key))
    replacement.type = replacement.__class__.__name__
    replacement.np = sum(p.numel() for p in replacement.parameters())
    model.model[index] = replacement


def build_model(variant="baseline", nc=4, weights=None, verbose=True, scale="n"):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if scale not in {"n", "s", "m"} or (variant != "baseline" and scale != "n"):
        raise ValueError("Use n/s/m for baseline; custom variants currently support n only")
    model = DetectionModel(f"yolov8{scale}.yaml", nc=nc, verbose=verbose)
    if variant in {"ghost", "pd"}:
        for index in (12, 15, 18, 21):
            layer = model.model[index]
            replace_layer(model, index, C2fGhost(layer.cv1.conv.in_channels,
                                               layer.cv2.conv.out_channels, len(layer.m)))
    if variant in {"bot", "pd"}:
        layer = model.model[8]
        replace_layer(model, 8, nn.Sequential(layer, BOT(layer.cv2.conv.out_channels)))
    if variant in {"lska", "pd"}:
        for index in (9, 21):
            layer = model.model[index]
            replace_layer(model, index, nn.Sequential(layer, LSKA(layer.cv2.conv.out_channels)))
    if variant in {"lscd", "pd"}:
        original = model.model[-1]
        channels = tuple(branch[0].conv.in_channels for branch in original.cv2)
        head = LSCDHead(nc=nc, ch=channels)
        head.stride = original.stride.clone()
        head.bias_init()
        replace_layer(model, len(model.model)-1, head)
    model.variant = variant
    if weights is not None:
        # Transfer only matching parameters; new modules start from random initialization.
        model.load(weights)
    return model


class RoadTrainer(DetectionTrainer):
    """Rebuild the custom architecture when Ultralytics starts training."""
    variant = "baseline"
    scale = "n"

    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(self.variant, self.data["nc"], weights, verbose, self.scale)
