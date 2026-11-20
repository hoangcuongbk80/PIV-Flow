import argparse
import csv
import json
from evaluation.statistics import paired_rows, hierarchical_interval


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Paired acquisition-bootstrap flow comparison")
    cli.add_argument("control"); cli.add_argument("method")
    cli.add_argument("--dataset", required=True, choices=["fcdn", "vbof"])
    cli.add_argument("--repeats", type=int, default=2000)
    args = cli.parse_args()
    def read(path):
        with open(path, newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    rows = paired_rows(read(args.control), read(args.method))
    hierarchy = ("base_pair_id",) if args.dataset == "fcdn" else ("group_id",)
    print(json.dumps(hierarchical_interval(rows, hierarchy=hierarchy, repeats=args.repeats,
                                          strata="camera_family" if args.dataset == "vbof" else None), indent=2))
