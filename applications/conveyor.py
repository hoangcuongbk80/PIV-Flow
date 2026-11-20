"""Perspective-aware physical speed, causal aggregation and operational references."""
from collections import deque
import numpy as np


def project(points, homography):
    flat = np.asarray(points, dtype=float).reshape(-1, 2)
    values = np.column_stack((flat, np.ones(len(flat)))) @ np.asarray(homography, dtype=float).T
    valid = np.isfinite(values).all(1) & (np.abs(values[:, 2]) > 1e-10)
    mapped = np.full((len(flat), 2), np.nan)
    mapped[valid] = values[valid, :2] / values[valid, 2:3]
    return mapped.reshape(np.asarray(points).shape)


def polygon_contains(points, polygon):
    """Vectorized ray casting, avoiding rasterization shifts at warped endpoints."""
    points = np.asarray(points)
    x, y = points[..., 0], points[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    polygon = np.asarray(polygon, dtype=float)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        dy = b[1] - a[1]
        if abs(dy) < 1e-12:
            continue
        crossing = ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / dy + a[0])
        inside ^= crossing
    return inside & np.isfinite(points).all(-1)


def outside_boxes(points, boxes):
    valid = np.ones(points.shape[:-1], dtype=bool)
    for box in np.asarray(boxes).reshape(-1, 4):
        x1, y1, x2, y2 = box
        dx, dy = .1 * (x2 - x1), .1 * (y2 - y1)
        valid &= ~((points[..., 0] >= x1 - dx) & (points[..., 0] <= x2 + dx)
                   & (points[..., 1] >= y1 - dy) & (points[..., 1] <= y2 + dy))
    return valid


def interval_speed(flow, dt, calibration, boxes1=(), boxes2=()):
    if dt <= 0 or not np.isfinite(dt):
        return None
    height, width = flow.shape[:2]
    y, x = np.mgrid[:height, :width]
    source = np.stack((x, y), -1).astype(float)
    target = source + flow
    polygon = calibration["belt_polygon"]
    static = polygon_contains(source, polygon)
    for region in calibration.get("exclusion_polygons", []):
        static &= ~polygon_contains(source, region)
    denominator = int(static.sum())
    valid = static & polygon_contains(target, polygon)
    for region in calibration.get("exclusion_polygons", []):
        valid &= ~polygon_contains(target, region)
    valid &= outside_boxes(source, boxes1) & outside_boxes(target, boxes2)
    h = np.asarray(calibration["homography"], dtype=float)
    direction = np.asarray(calibration["belt_direction"], dtype=float)
    length = np.linalg.norm(direction)
    if length <= 0 or h.shape != (3, 3):
        raise ValueError("Invalid belt direction or 3x3 homography.")
    displacement = project(target, h) - project(source, h)
    values = (displacement @ (direction / length)) / dt
    valid &= np.isfinite(values)
    speeds = values[valid]
    if not denominator or not len(speeds):
        return None
    median = np.median(speeds); mad = np.median(np.abs(speeds - median))
    speeds = speeds[np.abs(speeds - median) <= 3 * max(mad, 1e-3)]
    if len(speeds) < max(1, int(np.ceil(.01 * denominator))):
        return None
    return float(np.median(speeds))


class CausalSpeedWindow:
    def __init__(self, size=15, minimum=8):
        self.size, self.minimum = size, minimum
        self.predictions, self.references = deque(maxlen=size), deque(maxlen=size)

    def update(self, prediction, reference):
        self.predictions.append(prediction); self.references.append(reference)
        if len(self.predictions) < self.size:
            return {"warmup": True, "prediction": None, "reference": None}
        values = [v for v in self.predictions if v is not None and np.isfinite(v)]
        refs = list(self.references)
        return {"warmup": False,
                "prediction": float(np.median(values)) if len(values) >= self.minimum else None,
                "reference": float(np.median(refs)) if all(v is not None and np.isfinite(v) for v in refs) else None}


def reference_interval(rows, start, end, tolerance=1e-7):
    """Duration-weighted average; gaps/invalid intervals make the reference unavailable."""
    if end <= start:
        return None
    covered, weighted, cursor = 0., 0., start
    for row in rows:
        left, right = max(start, float(row["start_time_s"])), min(end, float(row["end_time_s"]))
        if right <= left:
            continue
        if left > cursor + tolerance or left < cursor - tolerance:
            return None  # missing support or overlapping operational measurements
        if str(row.get("valid", "1")).lower() not in ("1", "true"):
            return None
        speed = float(row["speed_mps"])
        if not np.isfinite(speed):
            return None
        covered += right - left; weighted += (right - left) * speed; cursor = right
    if abs(covered - (end - start)) > tolerance:
        return None
    return weighted / covered


def speed_metrics(rows):
    eligible = [r for r in rows if not r.get("warmup", False) and r.get("reference") is not None]
    valid = [r for r in eligible if r.get("prediction") is not None and np.isfinite(r["prediction"])]
    errors = np.asarray([r["prediction"] - r["reference"] for r in valid], dtype=float)
    return {"eligible": len(eligible), "valid": len(valid),
            "coverage_percent": 100 * len(valid) / len(eligible) if eligible else None,
            "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            "within_0_10_percent": float(100 * np.mean(np.abs(errors) <= .1)) if len(errors) else None}
