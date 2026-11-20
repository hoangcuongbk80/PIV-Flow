"""Photometric Intervention-Supervised Update Verification for optical flow."""
import torch
from torch import nn
from .extractor import Encoder
from .corr import CorrPyramid
from .geometry import pad_pair, convex_upsample
from .update import MotionEncoder, ConvGRU, prediction_head
from .verifier import UpdateVerifier, select_step


class PIVFlow(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        cfg = dict(model or {})
        size = cfg.get("size", "S").upper()
        width = cfg.get("width", 128 if size == "S" else 192)
        depth = 18 if size == "S" else 34
        self.iters = cfg.get("iters", 4)
        self.kind = cfg.get("verifier", "paired_quantile")
        self.policy = cfg.get("policy", "lower")
        self.margin = cfg.get("margin", 0.)
        self.endpoint_ablation = cfg.get("endpoint_ablation", False)
        self.radius = cfg.get("radius", 4)
        self.fnet = Encoder(depth, 3, width, cfg.get("pretrained", False))
        self.cnet = Encoder(depth, 6, width * 2, cfg.get("pretrained", False))
        self.init_head = prediction_head(width, 2)
        self.motion = MotionEncoder(width, 4 * (2 * self.radius + 1) ** 2)
        extra = cfg.get("proposal_adapter_hidden", 0)
        self.proposal_adapter = (nn.Sequential(nn.Conv2d(width, extra, 1), nn.ReLU(),
                                              nn.Conv2d(extra, width, 1)) if extra else None)
        self.gru = ConvGRU(width, width * 2 + 4)
        self.flow_head = prediction_head(width, 2)
        self.mask_head = prediction_head(width, 8 * 8 * 9)
        self.verifier = UpdateVerifier(width, width, self.kind) if cfg.get("use_verifier", True) else None
        if self.verifier is None and self.policy not in ("full", "half"):
            raise ValueError("A model without a verifier requires full or half policy.")
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def set_phase(self, phase):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("verifier.") if phase == "verifier"
                                     else (not name.startswith("verifier.") if phase == "warmup" else True))

    def forward(self, image1, image2, iters=None, policy=None, return_aux=False):
        if image1.shape != image2.shape or image1.ndim != 4 or image1.shape[1] != 3:
            raise ValueError("Expected equally sized N,3,H,W RGB tensors in [0,1].")
        policy = policy or self.policy
        image1, image2, (height, width) = pad_pair(image1, image2)
        image1 = (image1 - self.rgb_mean) / self.rgb_std
        image2 = (image2 - self.rgb_mean) / self.rgb_std
        f1, f2 = self.fnet(image1), self.fnet(image2)
        hidden, context = self.cnet(torch.cat((image1, image2), 1)).chunk(2, 1)
        hidden, context = hidden.tanh(), context.relu()
        corr = CorrPyramid(f1, f2)
        evidence = {"feature1": f1.detach(), "feature2": f2.detach(),
                    "context": context.detach(), "corr": corr}
        flow = self.init_head(hidden).float()
        feedback = flow.new_zeros(flow.shape[0], 4, *flow.shape[-2:])
        def upsample(value, mask):
            return convex_upsample(value, mask)[..., :height, :width]
        initial = upsample(flow, .25 * self.mask_head(hidden))
        committed, trials, traces = [], [], []
        for _ in range(self.iters if iters is None else iters):
            anchor = flow.detach()
            corr_features, _ = corr.lookup(anchor, self.radius)
            motion = self.motion(corr_features, anchor.to(corr_features.dtype))
            if self.proposal_adapter is not None:
                motion = motion + self.proposal_adapter(motion)
            hidden = self.gru(hidden, torch.cat((motion, context, feedback.to(context.dtype)), 1))
            delta = self.flow_head(hidden).float()
            half, full = anchor + .5 * delta, anchor + delta
            if policy in ("full", "half"):
                scale = torch.full_like(anchor[:, :1], 1. if policy == "full" else .5)
                action = (scale * 2).long()
                gains = anchor.new_zeros(anchor.shape[0], 2, *anchor.shape[-2:])
            else:
                # Selection has no gradient. Replay explicitly trains the verifier below.
                with torch.no_grad():
                    prepared = self.verifier.prepare(evidence)
                    signature = self.verifier.signature(evidence, prepared, anchor)
                    predictions = [self.verifier(evidence, anchor, candidate,
                                   self.endpoint_ablation, prepared, signature) for candidate in (half, full)]
                    action, scale, gains = select_step(predictions, delta, self.kind,
                                                       policy, self.margin)
            flow = anchor + scale * delta
            mask = .25 * self.mask_head(hidden)
            committed.append(upsample(flow, mask))
            trials.append(upsample(full, mask))
            feedback = torch.cat((flow - anchor, gains), 1).detach()
            if return_aux:
                traces.append({"anchor": anchor, "half": half.detach(), "full": full.detach(),
                               "action": action.detach(), "gains": gains.detach()})
        final = committed[-1] if committed else initial
        result = {"final": final, "initial": initial, "flow": committed, "trial": trials}
        if return_aux:
            result.update(evidence=evidence, traces=traces)
        return result
