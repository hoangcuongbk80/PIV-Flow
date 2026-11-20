"""Evaluate external-baseline .flo predictions on the same fixed manifest."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from core.datasets.io import read_manifest, read_flow
from applications.common import write_csv
import json


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True); cli.add_argument("--manifest", required=True)
    cli.add_argument("--predictions", required=True, help="CSV columns pair_id,prediction; paths relative to this CSV")
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    from applications.common import read_csv
    manifest = read_manifest(args.manifest); index = read_csv(args.predictions)
    predictions = {r["pair_id"]: r["prediction"] for r in index}
    if len(predictions) != len(index) or set(predictions) != {r["pair_id"] for r in manifest}:
        raise ValueError("External prediction IDs must match the manifest exactly, without duplicates.")
    rows = []
    for row in manifest:
        gt = read_flow(Path(args.root) / row["flow"])
        pred = read_flow(Path(args.predictions).parent / predictions[row["pair_id"]])
        if pred.shape != gt.shape or not np.isfinite(pred).all():
            raise ValueError("Prediction shape/nonfinite error; evaluation support cannot shrink.")
        valid = np.isfinite(gt).all(-1)
        if row.get("valid_mask"):
            with Image.open(Path(args.root) / row["valid_mask"]) as image:
                valid &= np.asarray(image.convert("L")) > 0
        if not valid.any():
            raise ValueError("Empty reference support.")
        output = {k: row.get(k, "") for k in ("pair_id", "base_pair_id", "group_id", "reference_id", "camera_family")}
        output["epe"] = float(np.linalg.norm(pred[valid] - gt[valid], axis=-1).mean())
        rows.append(output)
    target = Path(args.output); write_csv(target / "pairs.csv", rows)
    (target / "metrics.json").write_text(json.dumps({"pairs": len(rows), "epe": float(np.mean([r["epe"] for r in rows]))}, indent=2), encoding="utf-8")
