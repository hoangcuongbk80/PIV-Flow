import torch
from .geometry import feature_reference


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (torch.where(mask.bool(), values, torch.zeros_like(values)) * mask).sum() / mask.sum().clamp_min(1)


def flow_loss(prediction, reference, valid):
    if valid.ndim == 3:
        valid = valid[:, None]
    valid = valid.bool() & torch.isfinite(reference).all(1, keepdim=True)
    target = torch.where(valid, reference, torch.zeros_like(reference))
    return masked_mean((prediction.float() - target.float()).abs().mean(1, keepdim=True), valid)


def sequence_loss(output, reference, valid, gamma=.8, trial_weight=.5):
    count = len(output["flow"])
    loss = gamma ** count * flow_loss(output["initial"], reference, valid)
    for k, (flow, trial) in enumerate(zip(output["flow"], output["trial"])):
        loss = loss + gamma ** (count - k - 1) * (
            flow_loss(flow, reference, valid) + trial_weight * flow_loss(trial, reference, valid)) / (1 + trial_weight)
    return loss


def utility_target(anchor, trial, reference, valid, eps=1e-3):
    anchor, trial, reference = anchor.float(), trial.float(), reference.float()
    norm = (trial - anchor).norm(dim=1, keepdim=True)
    ea, eb = (anchor - reference).norm(dim=1, keepdim=True), (trial - reference).norm(dim=1, keepdim=True)
    gain = ea - eb
    eligible = valid.bool() & (norm >= eps)
    # Denominator is replaced before division; zero-length steps never divide by zero.
    denominator = torch.where(norm >= eps, norm, torch.ones_like(norm))
    target = torch.where(eligible, gain / denominator, torch.zeros_like(gain)).clamp(-1, 1)
    return target.detach(), gain.detach(), norm.detach(), eligible.detach(), ea.detach(), eb.detach()


def pinball(error, level):
    return torch.maximum(level * error, (level - 1) * error)


def verification_loss(prediction, targets, kind="paired_quantile", tau=.1):
    target, gain, norm, valid, ea, eb = targets
    if kind == "paired_quantile":
        return .5 * (masked_mean(pinball(target - prediction[:, :1], tau), valid)
                     + masked_mean(pinball(target - prediction[:, 1:2], .5), valid))
    if kind == "mean_gain":
        # Both channels receive the same mean target to keep head capacity matched.
        return masked_mean((prediction - target).square().mean(1, keepdim=True), valid)
    errors = torch.cat((ea, eb), 1)
    if kind == "marginal_quantile":
        return masked_mean(pinball(errors - prediction, 1 - tau).mean(1, keepdim=True), valid)
    if kind == "mean_error":
        return masked_mean((errors - prediction).square().mean(1, keepdim=True), valid)
    if kind == "confidence":
        labels = (gain > 0).float().expand_as(prediction)
        return masked_mean(torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, labels, reduction="none").mean(1, keepdim=True), valid)
    raise ValueError(f"Unsupported utility loss: {kind}")


def replay_loss(model, outputs, reference, valid, tau=.1, gamma=.8, cross_observation=True):
    """Four origin/trial pairs, each replayed under both observation evidences."""
    kmax = len(outputs[0]["traces"])
    weights = [gamma ** (kmax - k - 1) for k in range(kmax)]
    total = reference.new_zeros((), dtype=torch.float32)
    for k, weight in enumerate(weights):
        terms = []
        for origin in range(2):
            trace = outputs[origin]["traces"][k]
            gt, vf = feature_reference(reference, valid, trace["anchor"].shape[-2:])
            if model.kind == "action":
                observations = (0, 1) if cross_observation else (origin, origin)
                target_list = [utility_target(trace["anchor"], trace[key], gt, vf) for key in ("half", "full")]
                true_gains = torch.cat((torch.zeros_like(target_list[0][1]),
                                        target_list[0][1], target_list[1][1]), 1)
                eligible = torch.cat((target_list[0][3], target_list[1][3]), 1)
                true_gains[:, 1:] = torch.where(eligible, true_gains[:, 1:], torch.full_like(true_gains[:, 1:], -1e9))
                labels = true_gains.argmax(1)
                for observation in observations:
                    predicted = [model.verifier(outputs[observation]["evidence"], trace["anchor"], trace[key],
                                                model.endpoint_ablation).mean(1, keepdim=True) for key in ("half", "full")]
                    nonzero_logits = torch.cat(predicted, 1)
                    nonzero_logits = torch.where(eligible, nonzero_logits, torch.full_like(nonzero_logits, -1e9))
                    logits = torch.cat((torch.zeros_like(predicted[0]), nonzero_logits), 1)
                    error = torch.nn.functional.cross_entropy(logits, labels, reduction="none")[:, None]
                    terms.extend([masked_mean(error, vf & eligible.any(1, keepdim=True))] * 2)
                continue
            for candidate_name in ("half", "full"):
                anchor, trial = trace["anchor"], trace[candidate_name]
                targets = utility_target(anchor, trial, gt, vf)
                # Independent augmentation repeats its own evidence to match 8 comparisons.
                observations = (0, 1) if cross_observation else (origin, origin)
                for observation in observations:
                    pred = model.verifier(outputs[observation]["evidence"], anchor, trial,
                                          model.endpoint_ablation)
                    terms.append(verification_loss(pred, targets, model.kind, tau))
        total = total + weight / sum(weights) * torch.stack(terms).mean()
    return total
