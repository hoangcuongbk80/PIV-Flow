import argparse
import configparser
from pathlib import Path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Evaluate worker tracking using official TrackEval metrics")
    cli.add_argument("--gt", required=True); cli.add_argument("--trackers", required=True)
    cli.add_argument("--tracker", default="piv_flow"); cli.add_argument("--output", required=True)
    args = cli.parse_args()
    from evaluation.trackeval_compat import load_trackeval
    trackeval = load_trackeval()
    sequences = {}
    for path in sorted(Path(args.gt).glob("*/seqinfo.ini")):
        ini = configparser.ConfigParser(); ini.read(path)
        sequences[path.parent.name] = int(ini["Sequence"]["seqLength"])
    if not sequences:
        raise ValueError("No MOT sequence metadata found under --gt.")
    dataset_config = {"GT_FOLDER": str(Path(args.gt).resolve()), "TRACKERS_FOLDER": str(Path(args.trackers).resolve()),
                      "OUTPUT_FOLDER": str(Path(args.output).resolve()), "TRACKERS_TO_EVAL": [args.tracker],
                      "CLASSES_TO_EVAL": ["pedestrian"], "SEQ_INFO": sequences,
                      "SKIP_SPLIT_FOL": True, "DO_PREPROC": True, "BENCHMARK": "Underground", "SPLIT_TO_EVAL": "test"}
    evaluator = trackeval.Evaluator({"USE_PARALLEL": False, "PRINT_RESULTS": True, "OUTPUT_DETAILED": True})
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
