"""Validate an authoritative VBOF CSV index, then build camera-held-out folds."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from core.datasets.io import write_manifest


def prepare(rows, output, paper_protocol=False):
    fields = ("pair_id", "image1", "image2", "flow", "reference_id", "capture_id", "group_id", "camera_subset", "camera_family")
    families = {"Canon", "Fuji", "Nikon", "Sony"}
    ownership = defaultdict(set)
    for row in rows:
        for field in fields:
            if not row.get(field):
                raise ValueError(f"Missing authoritative VBOF field: {field}")
        if row["camera_family"] not in families:
            raise ValueError(f"Unknown camera family: {row['camera_family']}")
        for field in ("reference_id", "capture_id", "group_id"):
            ownership[(field, row[field])].add(row["camera_family"])
    if any(len(value) > 1 for value in ownership.values()):
        raise ValueError("Shared references/captures/groups cross camera families. Correct metadata before folding.")
    # Require a single acquisition group for any shared capture or directed reference.
    for field in ("reference_id", "capture_id"):
        groups = defaultdict(set)
        for row in rows:
            groups[row[field]].add(row["group_id"])
        if any(len(value) > 1 for value in groups.values()):
            raise ValueError(f"Shared {field} crosses acquisition groups; merge related groups first.")
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate pair_id.")
    stats = {"pairs": len(rows), "references": len({r["reference_id"] for r in rows}),
             "groups": len({r["group_id"] for r in rows}), "subsets": len({r["camera_subset"] for r in rows})}
    if paper_protocol and stats != {"pairs": 2349, "references": 426, "groups": 214, "subsets": 8}:
        raise ValueError(f"Index differs from the paper's audited manifest: {stats}")
    output = Path(output)
    write_manifest(output / "vbof_all.jsonl", rows)
    for family in sorted(families):
        test = [r for r in rows if r["camera_family"] == family]
        train = [dict(r) for r in rows if r["camera_family"] != family]
        for row in train:
            # Prevent accidental use of real labels during Mixed-U adaptation.
            row.pop("flow", None); row.pop("valid_mask", None)
        write_manifest(output / f"vbof_{family.lower()}_unlabeled.jsonl", train)
        write_manifest(output / f"vbof_{family.lower()}_test.jsonl", test)
    print(stats)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--index", required=True); cli.add_argument("--output", default="manifests")
    cli.add_argument("--paper-protocol", action="store_true")
    args = cli.parse_args()
    with open(args.index, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    prepare(rows, args.output, args.paper_protocol)
