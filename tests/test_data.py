import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from core.datasets.io import read_flow, write_flow, read_manifest, write_manifest
from tools.prepare_fcdn import prepare as prepare_fcdn
from tools.prepare_vbof import prepare as prepare_vbof
from tools.audit_data import audit
from tools.make_smoke_data import make


class DataTests(unittest.TestCase):
    def test_flow_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "flow.flo"
            flow = np.random.default_rng(1).normal(size=(7, 9, 2)).astype(np.float32)
            write_flow(path, flow)
            np.testing.assert_array_equal(read_flow(path), flow)
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                read_flow(path)

    def test_duplicate_manifest_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.jsonl"
            row = dict(pair_id="a", image1="a.png", image2="b.png")
            write_manifest(path, [row, row])
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_fcdn_grouped_reproducible_split(self):
        rows = []
        for base in range(8):
            for exposure in range(2):
                rows.append(dict(pair_id=f"{base}_{exposure}", base_pair_id=str(base), image1="a.png", image2="b.png",
                                 flow="a.flo", official_split="validation" if base == 7 else "training"))
        with tempfile.TemporaryDirectory() as folder:
            prepare_fcdn(rows, folder, development=2)
            train = read_manifest(Path(folder) / "fcdn_train.jsonl")
            dev = read_manifest(Path(folder) / "fcdn_dev.jsonl")
            self.assertEqual(len(train), 10); self.assertEqual(len(dev), 4)
            self.assertFalse({r["base_pair_id"] for r in train} & {r["base_pair_id"] for r in dev})

    def test_vbof_folds_remove_labels(self):
        rows = [dict(pair_id=str(i), image1="a", image2="b", flow="f", reference_id=f"r{i}", capture_id=f"c{i}",
                     group_id=f"g{i}", camera_subset=family + "1", camera_family=family)
                for i, family in enumerate(("Canon", "Fuji", "Nikon", "Sony"))]
        with tempfile.TemporaryDirectory() as folder:
            prepare_vbof(rows, folder)
            train = read_manifest(Path(folder) / "vbof_canon_unlabeled.jsonl")
            self.assertTrue(all("flow" not in row and row["camera_family"] != "Canon" for row in train))
            rows[1]["reference_id"] = rows[0]["reference_id"]
            with self.assertRaises(ValueError):
                prepare_vbof(rows, folder)

    def test_smoke_dataset_and_split_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            make(folder)
            root = Path(folder)
            result = audit(root, [root / "train.jsonl", root / "dev.jsonl"], check_pixels=True)
            self.assertEqual(sum(result.values()), 4)
            with self.assertRaises(ValueError):
                audit(root, [root / "train.jsonl", root / "train.jsonl"])
