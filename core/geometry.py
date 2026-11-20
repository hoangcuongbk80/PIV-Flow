"""Pixel coordinates are (x, y); low-resolution flow uses feature pixels."""
import torch
import torch.nn.functional as F


def coords_grid(batch, height, width, device, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((x, y), 0)[None].expand(batch, -1, -1, -1)


def sample_pixels(image, coords):
    """Bilinear zero sampling with align_corners=True; return a validity map."""
    h, w = image.shape[-2:]
    x, y = coords[:, 0], coords[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    grid = torch.stack((2 * x / max(w - 1, 1) - 1,
                        2 * y / max(h - 1, 1) - 1), -1)
    # Singleton axes must not silently turn arbitrary coordinates into valid samples.
    # Keep interpolation coordinates/reductions in float32 under CUDA autocast.
    value = F.grid_sample(image.float(), grid.float(), align_corners=True, padding_mode="zeros")
    value = torch.where(valid[:, None], value, torch.zeros_like(value))
    return value, valid[:, None]


def warp(image, flow):
    return sample_pixels(image, coords_grid(flow.shape[0], *flow.shape[-2:],
                                           flow.device) + flow.float())


def convex_upsample(flow, logits, stride=8):
    n, _, h, w = flow.shape
    weights = logits.float().view(n, 1, 9, stride, stride, h, w).softmax(2)
    neighbors = F.unfold(stride * flow.float(), 3, padding=1)
    neighbors = neighbors.view(n, 2, 9, 1, 1, h, w)
    up = (weights * neighbors).sum(2)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(n, 2, stride * h, stride * w)


def pad_pair(image1, image2, stride=8):
    h, w = image1.shape[-2:]
    ph, pw = (-h) % stride, (-w) % stride
    # Bottom/right padding keeps the reference sampling origin exactly at (0,0).
    return (F.pad(image1, (0, pw, 0, ph), mode="replicate"),
            F.pad(image2, (0, pw, 0, ph), mode="replicate"), (h, w))


def feature_reference(flow, valid, size, stride=8):
    """Sample integer source positions; never interpolate labels at boundaries."""
    h, w = flow.shape[-2:]
    hp, wp = size[0] * stride, size[1] * stride
    if valid.ndim == 3:
        valid = valid[:, None]
    finite = torch.isfinite(flow).all(1, keepdim=True)
    valid = valid.bool() & finite
    flow = torch.where(valid, flow, torch.zeros_like(flow))
    flow = F.pad(flow, (0, wp - w, 0, hp - h))
    valid = F.pad(valid.float(), (0, wp - w, 0, hp - h))
    return flow[:, :, ::stride, ::stride].float() / stride, valid[:, :, ::stride, ::stride].bool()
