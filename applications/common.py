import csv
import json
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def clip_frames(root, clip):
    """Clip frame IDs start at zero; timestamps are seconds on the reference clock."""
    import cv2
    root = Path(root)
    times = read_csv(root / clip["timestamps"])
    by_id = {int(r["frame_id"]): float(r["timestamp_s"]) for r in times}
    if len(by_id) != len(times):
        raise ValueError("Duplicate frame timestamp IDs.")
    crop = json.loads(clip.get("crop_xywh") or "null")
    def frame(rgb, index):
        if index not in by_id:
            raise ValueError(f"Missing timestamp for {clip['clip_id']} frame {index}")
        if index and by_id[index] <= by_id[index - 1]:
            raise ValueError("Frame timestamps must increase strictly.")
        if crop:
            x, y, w, h = map(int, crop)
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > rgb.shape[1] or y + h > rgb.shape[0]:
                raise ValueError("Crop is outside the original frame.")
            rgb = rgb[y:y + h, x:x + w]
        return index, by_id[index], rgb
    if clip.get("frames"):
        paths = sorted(p for p in (root / clip["frames"]).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        from core.datasets.io import read_image
        if len(paths) != len(by_id):
            raise ValueError("Frame files and timestamps differ in length.")
        for index, path in enumerate(paths):
            yield frame(read_image(path), index)
    else:
        capture = cv2.VideoCapture(str(root / clip["video"]))
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {clip['video']}")
        index = 0
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                yield frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), index)
                index += 1
        finally:
            capture.release()
        if index != len(by_id):
            raise ValueError("Decoded frames and timestamps differ in length.")


class Detector:
    def __init__(self, weights=None, device="cuda", cache=None, image_size=(480, 736)):
        self.device = (device.split(":", 1)[1] if ":" in device else "0") if device.startswith("cuda") else device
        self.image_size = image_size
        self.cache = None
        self.model = None
        if cache:
            self.cache = {}
            for row in read_csv(cache):
                key = (row["clip_id"], int(row["frame_id"]))
                self.cache.setdefault(key, [])
                # Explicit empty-frame rows have blank x1; absence is an error.
                if row.get("x1", "") != "":
                    self.cache[key].append([float(row[k]) for k in ("x1", "y1", "x2", "y2", "score")])
        else:
            if not weights:
                raise ValueError("Provide detector_weights or an explicit detection_cache.")
            from ultralytics import YOLO
            self.model = YOLO(weights)

    def __call__(self, rgb, clip_id, frame_id):
        if self.cache is not None:
            key = (clip_id, frame_id)
            if key not in self.cache:
                raise ValueError(f"Missing detection-cache frame: {key}")
            return np.asarray(self.cache[key], dtype=float).reshape(-1, 5)
        # Ultralytics NumPy inputs are BGR; convert RGB explicitly.
        result = self.model.predict(source=rgb[..., ::-1].copy(), device=self.device,
                                    conf=.35, iou=.6, classes=[0], imgsz=self.image_size,
                                    verbose=False)[0]
        return np.concatenate((result.boxes.xyxy.cpu().numpy(), result.boxes.conf[:, None].cpu().numpy()), 1)


class FlowPredictor:
    def __init__(self, model, device, amp=False):
        self.model, self.device, self.amp = model.eval(), device, amp

    def __call__(self, first, second):
        import torch
        images = [torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().to(self.device) / 255 for x in (first, second)]
        with torch.no_grad(), torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16,
                                             enabled=self.amp and self.device.startswith("cuda")):
            prediction = self.model(*images)["final"]
        result = prediction[0].permute(1, 2, 0).float().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite dense flow from the estimator.")
        return result
