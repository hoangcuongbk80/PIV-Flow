import argparse
import numpy as np
from applications.common import read_csv
from core.utils.checkpoint import save_json
from evaluation.statistics import hierarchical_interval


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Common-valid conveyor MAE differences with session/recording bootstrap")
    cli.add_argument("control"); cli.add_argument("method"); cli.add_argument("--output", required=True)
    args = cli.parse_args()
    def index(path):
        return {(r["clip_id"], r["frame_id"]): r for r in read_csv(path)
                if r["warmup"].lower() == "false" and r["reference"] != "" and r["prediction"] != ""}
    left, right = index(args.control), index(args.method)
    rows = []
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if abs(float(a["reference"]) - float(b["reference"])) > 1e-9:
            raise ValueError("Operational reference differs on a paired timestamp.")
        rows.append(dict(session_id=b["session_id"], recording_id=b["recording_id"],
                         delta=abs(float(a["prediction"]) - float(a["reference"]))
                               - abs(float(b["prediction"]) - float(b["reference"]))))
    result = hierarchical_interval(rows)
    result["common_valid_timestamps"] = len(rows)
    save_json(args.output, result)
    print(result)
