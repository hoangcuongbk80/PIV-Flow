"""Small generated translation pairs, exclusively for execution/gradient checks."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from core.datasets.io import write_flow, write_manifest


def make(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = []
    for index in range(4):
        first = rng.integers(0, 80, (64, 64, 3), dtype=np.uint8)
        second = np.roll(first, 2, axis=1)
        flow = np.zeros((64, 64, 2), np.float32); flow[..., 0] = 2.
        valid = np.full((64, 64), 255, np.uint8); valid[:, -2:] = 0
        Image.fromarray(first).save(root / f"{index}_img1.png")
        Image.fromarray(second).save(root / f"{index}_img2.png")
        Image.fromarray(valid).save(root / f"{index}_valid.png")
        write_flow(root / f"{index}.flo", flow)
        rows.append(dict(pair_id=str(index), base_pair_id=str(index), image1=f"{index}_img1.png", image2=f"{index}_img2.png",
                         flow=f"{index}.flo", valid_mask=f"{index}_valid.png", group_id=str(index)))
    write_manifest(root / "train.jsonl", rows[:3]); write_manifest(root / "dev.jsonl", rows[3:])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__); cli.add_argument("--output", default="datasets/smoke")
    make(cli.parse_args().output)
