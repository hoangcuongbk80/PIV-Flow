import argparse
import json
from pathlib import Path
from collections import Counter
from applications.common import read_csv


def audit(root, registry):
    root = Path(root); clips = read_csv(registry)
    identities, sessions, recordings = set(), {}, {}
    counts = Counter(); frames = Counter()
    for clip in clips:
        identity = clip["clip_id"]
        if identity in identities:
            raise ValueError(f"Duplicate clip_id: {identity}")
        identities.add(identity)
        task, partition = clip["task"], clip["split"]
        if task not in ("tracking", "conveyor"):
            raise ValueError("Unknown underground task.")
        if partition not in ({"training", "validation", "test"} if task == "tracking" else {"development", "test"}):
            raise ValueError("Unexpected task partition.")
        role = "development" if partition in ("validation", "development") else partition
        for key, owners in (("session_id", sessions), ("recording_id", recordings)):
            value = clip[key]
            if not value:
                raise ValueError(f"Missing {key}")
            if value in owners and owners[value] != role:
                raise ValueError(f"Acquisition leakage: {key}={value}")
            owners[value] = role
        if bool(clip.get("video")) == bool(clip.get("frames")):
            raise ValueError("Each clip needs exactly one video or frames path.")
        for key in ("video", "frames", "timestamps"):
            if clip.get(key) and not (root / clip[key]).exists():
                raise FileNotFoundError(root / clip[key])
        times = read_csv(root / clip["timestamps"])
        if [int(r["frame_id"]) for r in times] != list(range(len(times))):
            raise ValueError("Timestamp rows must list contiguous zero-based frame IDs.")
        stamps = [float(r["timestamp_s"]) for r in times]
        if any(b <= a for a, b in zip(stamps[:-1], stamps[1:])):
            raise ValueError("Non-increasing timestamps.")
        if task == "conveyor":
            for key in ("calibration", "reference"):
                if not (root / clip[key]).is_file():
                    raise FileNotFoundError(root / clip[key])
            reference = read_csv(root / clip["reference"])
            intervals = [(float(r["start_time_s"]), float(r["end_time_s"])) for r in reference]
            if any(b <= a for a, b in intervals) or any(b > c + 1e-7 for (_, b), (c, _) in zip(intervals[:-1], intervals[1:])):
                raise ValueError("Invalid/overlapping speed-reference intervals.")
        key = task + "/" + partition
        counts[key] += 1; frames[key] += len(times)
    return {"clips": dict(counts), "frames": dict(frames), "sessions": len(sessions), "recordings": len(recordings)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Validate private clip registries and session/recording disjointness")
    cli.add_argument("--root", required=True); cli.add_argument("--clips", required=True)
    args = cli.parse_args()
    print(json.dumps(audit(args.root, args.clips), indent=2))
