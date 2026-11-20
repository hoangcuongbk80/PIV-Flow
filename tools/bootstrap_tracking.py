"""Recompute nonlinear tracking metrics for paired session/recording resamples."""
import argparse
from collections import defaultdict
from pathlib import Path
import configparser
import numpy as np
from applications.common import read_csv
from core.utils.checkpoint import save_json


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--gt", required=True); cli.add_argument("--trackers", required=True)
    cli.add_argument("--control", nargs="+", required=True); cli.add_argument("--method", nargs="+", required=True)
    cli.add_argument("--clips", required=True); cli.add_argument("--repeats", type=int, default=2000)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    if len(args.control) != len(args.method):
        cli.error("Supply equal numbers of seed-matched tracker names.")
    from evaluation.trackeval_compat import load_trackeval
    trackeval = load_trackeval()
    clips = [r for r in read_csv(args.clips) if r["task"] == "tracking" and r["split"] == "test"]
    tree = defaultdict(lambda: defaultdict(list)); sequences = {}
    for clip in clips:
        name = clip["clip_id"]
        ini = configparser.ConfigParser(); ini.read(Path(args.gt) / name / "seqinfo.ini")
        sequences[name] = int(ini["Sequence"]["seqLength"])
        tree[clip["session_id"]][clip["recording_id"]].append(name)
    if not sequences:
        raise ValueError("No test tracking sequences.")
    dataset = trackeval.datasets.MotChallenge2DBox({"GT_FOLDER": str(Path(args.gt).resolve()),
        "TRACKERS_FOLDER": str(Path(args.trackers).resolve()), "TRACKERS_TO_EVAL": args.control + args.method,
        "SEQ_INFO": sequences, "SKIP_SPLIT_FOL": True, "CLASSES_TO_EVAL": ["pedestrian"], "DO_PREPROC": True})
    metric = trackeval.metrics.HOTA()
    cache = {}
    for tracker in set(args.control + args.method):
        cache[tracker] = {}
        for sequence in sequences:
            raw = dataset.get_raw_seq_data(tracker, sequence)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            cache[tracker][sequence] = metric.eval_sequence(data)
    def score(tracker, sampled):
        combined = metric.combine_sequences({str(i): cache[tracker][name] for i, name in enumerate(sampled)})
        return float(np.mean(combined["HOTA"]) * 100)
    def delta(sampled):
        return float(np.mean([score(b, sampled) - score(a, sampled) for a, b in zip(args.control, args.method)]))
    rng = np.random.default_rng(2026); sessions = sorted(tree); draws = []
    for _ in range(args.repeats):
        sampled = []
        for si in rng.integers(len(sessions), size=len(sessions)):
            recordings = tree[sessions[si]]; keys = sorted(recordings)
            for ri in rng.integers(len(keys), size=len(keys)):
                sampled.extend(recordings[keys[ri]])
        draws.append(delta(sampled))
    result = {"delta_hota_points": delta(list(sequences)), "ci95": np.quantile(draws, [.025, .975]).tolist(),
              "resamples": args.repeats, "seed_pairs": len(args.control), "sessions": len(tree),
              "interpretation": "Acquisition interval conditional on supplied trained seeds; raw HOTA components recombined."}
    save_json(args.output, result); print(result)
