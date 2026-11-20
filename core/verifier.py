import torch
from torch import nn
import torch.nn.functional as F
from .geometry import warp


class UpdateVerifier(nn.Module):
    def __init__(self, feature_dim, context_dim, kind="paired_quantile"):
        super().__init__()
        self.kind = kind
        self.projection = nn.Conv2d(feature_dim, 32, 1)
        # 3x32 feature channels + 36 correlations + 36 validity + 1 endpoint validity.
        self.endpoint = nn.Sequential(nn.Conv2d(169, 32, 1), nn.ReLU())
        self.context = nn.Sequential(nn.Conv2d(context_dim, 32, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(130, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 2, 1))

    def prepare(self, evidence):
        q1 = F.normalize(self.projection(evidence["feature1"].detach()), dim=1, eps=1e-6)
        q2 = F.normalize(self.projection(evidence["feature2"].detach()), dim=1, eps=1e-6)
        cv = self.context(evidence["context"].detach())
        return q1, q2, cv

    def signature(self, evidence, prepared, flow):
        q1, q2, _ = prepared
        warped, endpoint_valid = warp(q2, flow.detach())
        corr, valid = evidence["corr"].lookup(flow.detach(), radius=1, detach=True)
        signature = torch.cat((q1, warped, (q1 - warped).abs(), corr.detach(),
                               valid.to(q1.dtype), endpoint_valid.to(q1.dtype)), 1)
        return self.endpoint(signature)

    def forward(self, evidence, anchor, trial, endpoint_ablation=False,
                prepared=None, anchor_signature=None):
        prepared = self.prepare(evidence) if prepared is None else prepared
        ta = (self.signature(evidence, prepared, anchor) if anchor_signature is None
              else anchor_signature)
        tb = self.signature(evidence, prepared, trial)
        diff = tb - ta
        if endpoint_ablation:
            tb, diff = torch.zeros_like(tb), torch.zeros_like(diff)
        logits = self.head(torch.cat((ta, tb, diff, (trial - anchor).detach(), prepared[2]), 1))
        if self.kind == "paired_quantile":
            lo = logits[:, :1].float().tanh()
            median = lo + (1 - lo) * logits[:, 1:2].float().sigmoid()
            return torch.cat((lo, median), 1)
        if self.kind == "mean_gain":
            return logits.float().tanh()
        if self.kind in ("marginal_quantile", "mean_error"):
            return F.softplus(logits.float())
        if self.kind in ("confidence", "action"):
            return logits.float()  # binary candidate utility or a candidate action logit.
        raise ValueError(f"Unknown verifier: {self.kind}")


def select_step(predictions, delta, kind="paired_quantile", policy="lower", margin=0., eps=1e-3):
    """Return action indices, applied scale, finite unpenalized gain feedback."""
    full_norm = delta.float().norm(dim=1, keepdim=True)
    norms = torch.cat((full_norm * .5, full_norm), 1)
    if kind == "paired_quantile":
        channel = 1 if policy == "median" else 0
        gains = torch.cat([p[:, channel:channel + 1] for p in predictions], 1) * norms
    elif kind == "mean_gain":
        gains = torch.cat([p[:, :1] for p in predictions], 1) * norms
    elif kind in ("marginal_quantile", "mean_error"):
        gains = torch.cat([p[:, :1] - p[:, 1:2] for p in predictions], 1)
    elif kind == "action":
        gains = torch.cat([p.mean(1, keepdim=True) for p in predictions], 1)
    else:
        gains = torch.cat([p[:, :1] for p in predictions], 1)
    eligible = norms >= eps
    finite = torch.isfinite(gains)
    feedback = torch.where(eligible & finite, gains, torch.zeros_like(gains))
    scores = torch.where(eligible & finite, gains - margin, torch.full_like(gains, -torch.inf))
    zero = torch.zeros_like(full_norm)
    if policy == "no_zero":
        zero = torch.where(eligible.any(1, keepdim=True), torch.full_like(zero, -torch.inf), zero)
    actions = torch.cat((zero, scores), 1).argmax(1, keepdim=True)
    # torch.argmax picks the first maximum: zero, then half, then full.
    scale = actions.float() / 2
    return actions, scale, feedback
