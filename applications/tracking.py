"""Fixed translation-only worker tracker; no learned verifier scores are consumed."""
from dataclasses import dataclass
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    def linear_sum_assignment(cost):
        """Rectangular Hungarian fallback; SciPy remains the preferred fast path."""
        cost = np.asarray(cost, dtype=float)
        transposed = cost.shape[0] > cost.shape[1]
        if transposed:
            cost = cost.T
        n, m = cost.shape
        u, v = np.zeros(n + 1), np.zeros(m + 1)
        p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
        for i in range(1, n + 1):
            p[0] = i; j0 = 0
            minimum, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
            while True:
                used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
                for j in range(1, m + 1):
                    if not used[j]:
                        current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                        if current < minimum[j]:
                            minimum[j], way[j] = current, j0
                        if minimum[j] < delta:
                            delta, j1 = minimum[j], j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta; v[j] -= delta
                    else:
                        minimum[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
                if j0 == 0:
                    break
        cols = np.array([j - 1 for j in range(1, m + 1) if p[j] != 0], dtype=int)
        rows = np.array([p[j + 1] - 1 for j in cols], dtype=int)
        return (cols, rows) if transposed else (rows, cols)


def iou_matrix(first, second):
    first, second = np.asarray(first).reshape(-1, 4), np.asarray(second).reshape(-1, 4)
    low = np.maximum(first[:, None, :2], second[None, :, :2])
    high = np.minimum(first[:, None, 2:], second[None, :, 2:])
    overlap = np.maximum(high - low, 0).prod(-1)
    area1 = np.maximum(first[:, 2:] - first[:, :2], 0).prod(-1)
    area2 = np.maximum(second[:, 2:] - second[:, :2], 0).prod(-1)
    return overlap / np.maximum(area1[:, None] + area2[None] - overlap, 1e-12)


def robust_displacement(flow, box):
    height, width = flow.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    dx, dy = .1 * (x2 - x1), .1 * (y2 - y1)
    left, top = max(0, int(np.ceil(x1 + dx))), max(0, int(np.ceil(y1 + dy)))
    right, bottom = min(width, int(np.ceil(x2 - dx))), min(height, int(np.ceil(y2 - dy)))
    total = max(0, right - left) * max(0, bottom - top)
    if not total:
        return None
    values = flow[top:bottom, left:right].reshape(-1, 2)
    values = values[np.isfinite(values).all(1)]
    if not len(values):
        return None
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    values = values[(np.abs(values - median) <= 3 * np.maximum(mad, 1e-3)).all(1)]
    if len(values) < max(1, int(np.ceil(.05 * total))):
        return None
    return np.median(values, axis=0)


@dataclass
class Track:
    identity: int
    box: np.ndarray
    score: float
    streak: int = 1
    missed: int = 0
    confirmed: bool = False
    last_displacement: object = None


class WorkerTracker:
    def __init__(self, iou_threshold=.3, max_age=15, confirm_hits=3):
        self.threshold, self.max_age, self.confirm_hits = iou_threshold, max_age, confirm_hits
        self.tracks, self.next_id = [], 1

    def update(self, detections, flow=None):
        detections = np.asarray(detections, dtype=float).reshape(-1, 5)
        predicted = []
        for track in self.tracks:
            displacement = np.zeros(2)
            if flow is not None:
                value = robust_displacement(flow, track.box)
                if value is not None:
                    track.last_displacement = value
                if track.last_displacement is not None:
                    displacement = track.last_displacement
            predicted.append(track.box + np.tile(displacement, 2))
        matched_tracks, matched_detections = set(), set()
        if predicted and len(detections):
            overlap = iou_matrix(predicted, detections[:, :4])
            cost = np.where(overlap >= self.threshold, 1 - overlap, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if overlap[i, j] < self.threshold:
                    continue
                track = self.tracks[i]
                track.box, track.score = detections[j, :4].copy(), float(detections[j, 4])
                track.missed = 0; track.streak += 1
                track.confirmed |= track.streak >= self.confirm_hits
                matched_tracks.add(i); matched_detections.add(j)
        retained = []
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.box = predicted[i]; track.missed += 1; track.streak = 0
            if (i in matched_tracks or track.confirmed) and track.missed <= self.max_age:
                retained.append(track)
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                retained.append(Track(self.next_id, detection[:4].copy(), float(detection[4]),
                                      confirmed=self.confirm_hits <= 1))
                self.next_id += 1
        self.tracks = retained
        return [track for track in self.tracks if track.confirmed]
