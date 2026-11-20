"""Index released FCDN pairs and derive the paper's internal grouped split.

No synthetic degradation is generated here. An explicit release index takes
precedence over the optional FlyingChairs-compatible filename adapter.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from core.datasets.io import write_manifest


def conventional_index(root, split_path):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".ppm")
                   and p.stem.endswith("_img1"))
    if not files:
        raise ValueError("No *_img1 image pairs found. Supply --index with the release's true pairing.")
    entries = []
    for first in files:
        base = first.stem[:-5]
        seconds = [p for p in first.parent.glob(base + "_img2.*") if p.suffix.lower() in (".png", ".jpg", ".ppm")]
        flows = [p for pattern in (base + "_flow.flo", base + ".flo") for p in first.parent.glob(pattern)]
        if len(seconds) != 1 or len(flows) != 1:
            raise ValueError(f"Ambiguous/missing image2 or reference for {first}; use an explicit index.")
        entries.append({"pair_id": str(first.relative_to(root).with_suffix("")), "base_pair_id": base,
                        "image1": first.relative_to(root).as_posix(), "image2": seconds[0].relative_to(root).as_posix(),
                        "flow": flows[0].relative_to(root).as_posix()})
    ids = sorted(set(e["base_pair_id"] for e in entries))
    split = np.loadtxt(split_path, dtype=int).reshape(-1)
    if len(split) != len(ids):
        raise ValueError("Official split list must align with unique sorted base IDs; use --index otherwise.")
    mapping = dict(zip(ids, split))
    for entry in entries:
        value = mapping[entry["base_pair_id"]]
        if value not in (1, 2):
            raise ValueError("Official FlyingChairs split values must be 1 or 2.")
        entry["official_split"] = "training" if value == 1 else "validation"
    return entries


def prepare(entries, output, development=2224, seed=2026, paper_protocol=False):
    for row in entries:
        for field in ("pair_id", "base_pair_id", "image1", "image2", "flow", "official_split"):
            if not row.get(field):
                raise ValueError(f"Missing {field} in FCDN index.")
        if row["official_split"] not in ("training", "validation"):
            raise ValueError("official_split must be training or validation.")
    training = sorted({row["base_pair_id"] for row in entries if row["official_split"] == "training"})
    validation = {row["base_pair_id"] for row in entries if row["official_split"] == "validation"}
    if set(training) & validation:
        raise ValueError("A base pair crosses official training/validation.")
    if not 0 < development < len(training):
        raise ValueError("Internal development size must be smaller than official training base-pair count.")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(training))
    dev_ids = {training[i] for i in permutation[:development]}
    parts = {"train": [], "dev": [], "validation": []}
    for row in sorted(entries, key=lambda r: r["pair_id"]):
        name = "validation" if row["official_split"] == "validation" else ("dev" if row["base_pair_id"] in dev_ids else "train")
        row = dict(row, group_id=row["base_pair_id"])
        parts[name].append(row)
    counts = {key: len(value) for key, value in parts.items()}
    if paper_protocol and counts != {"train": 20008, "dev": 2224, "validation": 640}:
        raise ValueError(f"Release does not match paper counts: {counts}")
    for name, rows in parts.items():
        write_manifest(Path(output) / ("fcdn_" + name + ".jsonl"), rows)
    print(counts)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True); cli.add_argument("--index")
    cli.add_argument("--official-split"); cli.add_argument("--output", default="manifests")
    cli.add_argument("--dev-size", type=int, default=2224); cli.add_argument("--seed", type=int, default=2026)
    cli.add_argument("--paper-protocol", action="store_true")
    args = cli.parse_args()
    if args.index:
        with open(args.index, newline="", encoding="utf-8-sig") as stream:
            entries = list(csv.DictReader(stream))
    else:
        if not args.official_split:
            cli.error("Use --index or provide --official-split for the conventional filename adapter.")
        entries = conventional_index(Path(args.root), args.official_split)
    prepare(entries, args.output, args.dev_size, args.seed, args.paper_protocol)
