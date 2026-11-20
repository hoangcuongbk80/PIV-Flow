import csv
import json
import tempfile
import unittest
from pathlib import Path
from applications.common import write_csv, read_csv, Detector
from tools.audit_underground import audit
from tools.export_tracking_gt import export
from evaluation.trackeval_compat import _NumpyView


class RegistryTests(unittest.TestCase):
    def test_clip_registry_and_tracking_coordinate_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "frames").mkdir()
            write_csv(root / "timestamps.csv", [{"frame_id": i, "timestamp_s": i / 30} for i in range(3)])
            clips = [dict(clip_id="worker1", task="tracking", split="test", session_id="S07", recording_id="R01",
                          video="", frames="frames", timestamps="timestamps.csv", crop_xywh="[100,100,736,480]")]
            write_csv(root / "clips.csv", clips)
            self.assertEqual(audit(root, root / "clips.csv")["clips"]["tracking/test"], 1)
            annotations = [dict(clip_id="worker1", frame_id=0, track_id=1, x=120, y=140, width=70, height=190, ignore=0)]
            write_csv(root / "tracks.csv", annotations)
            export(root, root / "clips.csv", root / "tracks.csv", root / "mot")
            first = (root / "mot" / "worker1" / "gt" / "gt.txt").read_text().strip().split(",")
            self.assertEqual(first[:4], ["1", "1", "21.0", "41.0"])
            clips.append(dict(clips[0], clip_id="worker2", split="training"))
            write_csv(root / "clips.csv", clips)
            with self.assertRaises(ValueError):
                audit(root, root / "clips.csv")

    def test_detection_cache_empty_vs_missing_frame(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detections.csv"
            write_csv(path, [dict(clip_id="c", frame_id=0, x1="", y1="", x2="", y2="", score="")])
            detector = Detector(cache=path)
            self.assertEqual(detector(np.zeros((4, 4, 3)), "c", 0).shape, (0, 5))
            with self.assertRaises(ValueError):
                detector(np.zeros((4, 4, 3)), "c", 1)

    def test_scoped_numpy_compatibility(self):
        import numpy as np
        view = _NumpyView()
        self.assertIs(view.int, int)
        self.assertIs(view.float32, np.float32)
