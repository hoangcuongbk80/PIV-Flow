import numpy as np


def flow_rgb(flow, max_magnitude=None):
    """HSV flow visualization, white at zero; reference/prediction can share a scale."""
    import cv2
    flow = np.asarray(flow, dtype=np.float32)
    flow = np.where(np.isfinite(flow), flow, 0)
    magnitude = np.linalg.norm(flow, axis=-1)
    maximum = max(float(np.max(magnitude)) if max_magnitude is None else max_magnitude, 1e-6)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hsv = np.stack(((angle + np.pi) / (2 * np.pi) * 179,
                    np.clip(magnitude / maximum, 0, 1) * 255, np.full_like(angle, 255)), -1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
