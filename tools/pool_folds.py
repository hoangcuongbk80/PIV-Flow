import argparse
from pathlib import Path
import numpy as np
from applications.common import read_csv, write_csv
import json


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Pool disjoint out-of-fold VBOF predictions, pair-weighted")
    cli.add_argument("--folds", nargs="+", required=True); cli.add_argument("--output", required=True)
    args = cli.parse_args()
    rows, seen = [], set()
    for path in args.folds:
        fold = read_csv(path)
        keys = {r["pair_id"] for r in fold}
        if len(keys) != len(fold) or keys & seen:
            raise ValueError("Fold pairs overlap or contain duplicate predictions.")
        rows.extend(fold); seen.update(keys)
    if not rows:
        raise ValueError("No fold predictions.")
    target = Path(args.output); write_csv(target / "pairs.csv", rows)
    (target / "metrics.json").write_text(json.dumps({"pairs": len(rows), "epe": float(np.mean([float(r["epe"]) for r in rows])),
        "folds": args.folds}, indent=2), encoding="utf-8")
