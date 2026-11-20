import argparse
import csv
import numpy as np
from core.utils.checkpoint import save_json


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Seed means/SDs and seed-averaged pair rows for acquisition bootstrap")
    cli.add_argument("--runs", nargs="+", required=True); cli.add_argument("--output", required=True)
    args = cli.parse_args()
    runs = []
    for path in args.runs:
        with open(path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        index = {r["pair_id"]: r for r in rows}
        if len(index) != len(rows):
            raise ValueError("Duplicate pair IDs.")
        runs.append(index)
    if any(run.keys() != runs[0].keys() for run in runs):
        raise ValueError("Seed runs must have identical fixed evaluation supports.")
    rows = []
    for key in sorted(runs[0]):
        row = dict(runs[0][key]); row.pop("prediction", None)
        row["epe"] = float(np.mean([float(run[key]["epe"]) for run in runs]))
        for name in ("epe_dark", "epe_remaining"):
            if all(run[key].get(name, "") != "" for run in runs):
                row[name] = float(np.mean([float(run[key][name]) for run in runs]))
        rows.append(row)
    from pathlib import Path
    target = Path(args.output); target.mkdir(parents=True, exist_ok=True)
    with (target / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    means = [float(np.mean([float(row["epe"]) for row in run.values()])) for run in runs]
    save_json(target / "metrics.json", {"epe_mean": float(np.mean(means)), "seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                                        "seed_epes": means, "pairs": len(rows)})
