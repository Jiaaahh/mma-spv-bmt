"""Inference-only paths. Caller must verify frozen source hashes before import."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import Encoder
from budget_adapter import render
from vendor.resnet_arch import ResNet, Bottleneck
from hog_pipeline import Adapter, norm

PREFIXES = ('encoder.', 'channel.', 'blind_feat.', 'classifier.')

def patch(glyph, point):
    y, x = map(int, point)
    if glyph.shape != (96, 96) or glyph.dtype != np.float32 or not (0 <= y < 96 and 0 <= x < 96):
        raise ValueError('glyph/point contract')
    crop = np.pad(glyph, 40)[y:y+80, x:x+80].copy()
    yy, xx = np.mgrid[:80, :80]
    crop *= (yy-40)**2 + (xx-40)**2 <= 40**2
    return F.interpolate(torch.from_numpy(crop)[None, None], size=(128, 128), mode='bilinear', align_corners=False)[0].repeat(3, 1, 1)

class PointInference(nn.Module):
    def __init__(self):
        super().__init__()
        blind = ResNet(Bottleneck, [3, 8, 36, 3])
        self.encoder = Encoder()
        self.channel = nn.Conv2d(1, 3, 3, padding=1)
        self.blind_feat = nn.Sequential(*list(blind.children())[:6])
        self.classifier = nn.Sequential(*list(blind.children())[6:-1], nn.Flatten(), blind.fc, nn.Linear(1000, 180))

    def forward(self, x, kernels):
        stimulation = self.encoder(x)
        spv = render(stimulation, kernels)
        vision = F.interpolate(spv, size=(128, 128), mode='bilinear', align_corners=False)
        logits = self.classifier(self.blind_feat(self.channel(vision)))
        return logits, stimulation, spv

    def predict(self, glyph, point, kernels):
        device = next(self.parameters()).device
        x = patch(glyph, point)[None].to(device)
        with torch.inference_mode():
            return int(self(x, kernels)[0].argmax(1).cpu().item())

class HogInference:
    def __init__(self, bank_path):
        self.bank = np.load(bank_path, allow_pickle=False)
        if self.bank.shape != (180, 4356) or self.bank.dtype != np.float64:
            raise ValueError('prototype bank contract')
        self.adapter = Adapter('skeleton')

    def details(self, glyph, point, kernel):
        selected = self.adapter.select(patch(glyph, point))
        feature = self.adapter.feature(selected, kernel)
        scores = norm(feature) @ self.bank.T
        return scores, selected, feature

    def predict(self, glyph, point, kernel):
        return int(self.details(glyph, point, kernel)[0].argmax())
