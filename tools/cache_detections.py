import argparse
from pathlib import Path
from applications.common import read_csv, write_csv, clip_frames, Detector


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Cache exactly the same crop-coordinate detections for every flow method")
    cli.add_argument("--root", required=True); cli.add_argument("--clips", required=True)
    cli.add_argument("--weights", required=True); cli.add_argument("--device", default="cuda")
    cli.add_argument("--output", required=True); cli.add_argument("--split", default="test")
    args = cli.parse_args()
    detector = Detector(args.weights, args.device)
    rows = []
    for clip in read_csv(args.clips):
        if clip["split"] != args.split:
            continue
        for frame_id, _, rgb in clip_frames(args.root, clip):
            values = detector(rgb, clip["clip_id"], frame_id)
            for box in values:
                rows.append(dict(clip_id=clip["clip_id"], frame_id=frame_id,
                                 **dict(zip(("x1", "y1", "x2", "y2", "score"), box.tolist()))))
            if not len(values):
                rows.append(dict(clip_id=clip["clip_id"], frame_id=frame_id, x1="", y1="", x2="", y2="", score=""))
    write_csv(args.output, rows)
