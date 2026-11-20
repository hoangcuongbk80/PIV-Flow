"""Per-pair/per-iteration utility diagnostics, with and without policy selection."""
import torch
from core.geometry import feature_reference
from core.losses import utility_target
from core.verifier import select_step


@torch.no_grad()
def diagnose(model, evidence, trace, gt, valid, locations=None, policy="lower"):
    anchor, half, full = [trace[k] for k in ("anchor", "half", "full")]
    reference, vf = feature_reference(gt, valid, anchor.shape[-2:])
    targets = [utility_target(anchor, trial, reference, vf) for trial in (half, full)]
    predictions = [model.verifier(evidence, anchor, trial, model.endpoint_ablation) for trial in (half, full)]
    action, _, _ = select_step(predictions, full - anchor, model.kind, policy, model.margin)
    gains = torch.cat((torch.zeros_like(targets[0][1]), targets[0][1], targets[1][1]), 1)
    eligible = torch.cat((targets[0][3], targets[1][3]), 1)
    eligible_gains = torch.cat((gains[:, :1], torch.where(eligible, gains[:, 1:], torch.full_like(gains[:, 1:], -torch.inf))), 1)
    support = vf & eligible.any(1, keepdim=True)
    if locations is not None:
        support = support & locations
    if not support.any():
        return None
    selected = gains.gather(1, action)
    accepted = (action > 0) & support
    useful = (eligible_gains.max(1, keepdim=True).values > 0) & support
    def mean(value, mask):
        return value[mask].float().mean().item() if mask.any() else None
    result = {"acceptance": mean(action > 0, support), "harmful_acceptance": mean(selected < 0, accepted),
              "useful_rejection": mean((action == 0) & useful, support),
              "regret": mean(eligible_gains.max(1, keepdim=True).values - selected, support),
              "selected_gain": mean(selected, support), "oracle_gain": mean(eligible_gains.max(1, keepdim=True).values, support),
              "locations": support.sum().item()}
    if model.kind == "paired_quantile":
        tails, selected_tails = [], []
        for j, (prediction, target) in enumerate(zip(predictions, targets), 1):
            mask = target[3] & support
            tails.append((target[0] < prediction[:, :1], mask))
            selected_tails.append((target[0] < prediction[:, :1], mask & (action == j)))
        def pooled(items):
            count = sum(mask.sum().item() for _, mask in items)
            return sum(value[mask].float().sum().item() for value, mask in items) / count if count else None
        result.update(lower_tail_before=pooled(tails), lower_tail_selected=pooled(selected_tails))
    return result
