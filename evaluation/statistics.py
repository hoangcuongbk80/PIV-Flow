"""Paired, acquisition-aware bootstrap utilities (NumPy only)."""
from collections import defaultdict
import numpy as np


def hierarchical_interval(records, value_key="delta", hierarchy=("session_id", "recording_id"),
                          repeats=2000, seed=2026, strata=None):
    """Resample nested clusters, preserving all rows in selected terminal clusters.

    The statistic is a row-weighted mean after hierarchical resampling. For pair-
    equal flow EPE each row is one pair; tracking can supply threshold-level raw
    counts to an application-specific reducer instead (see bootstrap_tracking.py).
    """
    if not records:
        raise ValueError("Cannot bootstrap empty paired support.")
    rng = np.random.default_rng(seed)
    def build(rows, levels):
        if not levels:
            values = np.array([float(row[value_key]) for row in rows])
            if not np.isfinite(values).all():
                raise ValueError("Bootstrap values must be finite.")
            return float(values.sum()), len(values)
        groups = defaultdict(list)
        for row in rows:
            if not row.get(levels[0]):
                raise ValueError(f"Missing bootstrap grouping key {levels[0]}")
            groups[str(row[levels[0]])].append(row)
        return [build(groups[key], levels[1:]) for key in sorted(groups)]
    def sample(node):
        if isinstance(node, tuple):
            return node
        total, count = 0., 0
        for index in rng.integers(0, len(node), len(node)):
            subtotal, subcount = sample(node[index])
            total += subtotal; count += subcount
        return total, count
    if strata:
        grouped = defaultdict(list)
        for row in records:
            grouped[row[strata]].append(row)
        populations = list(grouped.values())
    else:
        populations = [records]
    trees = [build(population, hierarchy) for population in populations]
    draws = []
    for _ in range(repeats):
        total, count = 0., 0
        for tree in trees:
            subtotal, subcount = sample(tree)
            total += subtotal; count += subcount
        draws.append(total / count)
    return {"mean": float(np.mean([r[value_key] for r in records])),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "resamples": repeats}


def paired_rows(first, second, key="pair_id", value="epe"):
    left, right = {r[key]: r for r in first}, {r[key]: r for r in second}
    if left.keys() != right.keys():
        raise ValueError("Paired flow comparison requires identical pair supports.")
    return [dict(right[k], delta=float(left[k][value]) - float(right[k][value])) for k in sorted(left)]
