import argparse
import json
from pathlib import Path
import numpy as np
from core.datasets.io import read_manifest, read_image, read_flow


def audit(root, paths, check_pixels=False):
    seen = {key: {} for key in ("pair_id", "base_pair_id", "group_id", "reference_id", "capture_id")}
    result = {}
    for partition, path in enumerate(paths):
        rows = read_manifest(path)
        result[str(path)] = len(rows)
        for row in rows:
            for key, owners in seen.items():
                value = row.get(key)
                if value:
                    if value in owners and owners[value] != partition:
                        raise ValueError(f"Split leakage: {key}={value}")
                    owners[value] = partition
            for key in ("image1", "image2", "flow", "valid_mask"):
                if row.get(key) and not (Path(root) / row[key]).is_file():
                    raise FileNotFoundError(Path(root) / row[key])
            if check_pixels:
                a, b = read_image(Path(root) / row["image1"]), read_image(Path(root) / row["image2"])
                if a.shape != b.shape:
                    raise ValueError(f"Pair shape mismatch: {row['pair_id']}")
                if row.get("flow"):
                    flow = read_flow(Path(root) / row["flow"])
                    if flow.shape[:2] != a.shape[:2] or not np.isfinite(flow).all(-1).any():
                        raise ValueError(f"Invalid reference: {row['pair_id']}")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Check file support and disjoint acquisition identities")
    cli.add_argument("--root", required=True); cli.add_argument("--manifests", nargs="+", required=True)
    cli.add_argument("--pixels", action="store_true")
    args = cli.parse_args()
    print(json.dumps(audit(args.root, args.manifests, args.pixels), indent=2))
