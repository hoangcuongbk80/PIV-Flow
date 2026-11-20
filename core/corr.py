"""Differentiable all-pairs correlation with a fixed source grid."""
import math
import torch
import torch.nn.functional as F
from .geometry import sample_pixels, coords_grid


class CorrPyramid:
    def __init__(self, feature1, feature2, levels=4):
        n, c, h, w = feature1.shape
        self.shape = (n, h, w)
        corr = torch.matmul(feature1.flatten(2).transpose(1, 2), feature2.flatten(2))
        corr = corr / math.sqrt(c)
        corr = corr.reshape(n * h * w, 1, h, w)
        self.levels = [corr]
        for _ in range(levels - 1):
            if min(corr.shape[-2:]) < 2:
                raise ValueError("Four pyramid levels require padded input dimensions >=64.")
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.levels.append(corr)

    def lookup(self, flow, radius, detach=False):
        n, h, w = self.shape
        centers = coords_grid(n, h, w, flow.device) + flow.float()
        centers = centers.permute(0, 2, 3, 1).reshape(n * h * w, 2, 1, 1)
        oy, ox = torch.meshgrid(torch.arange(-radius, radius + 1, device=flow.device),
                               torch.arange(-radius, radius + 1, device=flow.device), indexing="ij")
        offsets = torch.stack((ox, oy), 0).float()[None]
        values, validities = [], []
        for level, volume in enumerate(self.levels):
            if detach:
                volume = volume.detach()
            samples, mask = sample_pixels(volume, centers / 2 ** level + offsets)
            values.append(samples.reshape(n, h, w, -1).permute(0, 3, 1, 2))
            validities.append(mask.reshape(n, h, w, -1).permute(0, 3, 1, 2))
        return torch.cat(values, 1), torch.cat(validities, 1)
