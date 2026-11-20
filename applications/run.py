"""Run with python -m applications.run --task tracking|conveyor ..."""
import json
from pathlib import Path
from core.utils.config import parser, load_config
from core.utils.checkpoint import load_model, save_json
from .common import read_csv, write_csv, clip_frames, Detector, FlowPredictor
from .tracking import WorkerTracker
from .conveyor import interval_speed, reference_interval, CausalSpeedWindow, speed_metrics


def main(args):
    cfg = load_config(args.config); application = cfg["application"]
    root = Path(application["root"])
    clips = [r for r in read_csv(root / application["clips"]) if r["task"] == args.task and r["split"] == args.split]
    if not clips:
        raise ValueError("No clips match task and split.")
    cache = application.get("detection_cache")
    detector = Detector(application.get("detector_weights"), args.device,
                        root / cache if cache else None)
    predict = None if args.detector_only else FlowPredictor(load_model(cfg, args.checkpoint, args.device), args.device, args.amp)
    output = Path(args.output); all_speeds = []
    for clip in clips:
        tracker, window = WorkerTracker(), CausalSpeedWindow()
        previous, previous_boxes, previous_time = None, None, None
        mot_rows, speed_rows = [], []
        if args.task == "conveyor":
            calibration = json.loads((root / clip["calibration"]).read_text(encoding="utf-8"))
            references = sorted(read_csv(root / clip["reference"]), key=lambda r: float(r["start_time_s"]))
        for frame_id, timestamp, rgb in clip_frames(root, clip):
            detections = detector(rgb, clip["clip_id"], frame_id)
            flow = predict(previous, rgb) if previous is not None and predict else None
            if args.task == "tracking":
                for track in tracker.update(detections, flow):
                    x1, y1, x2, y2 = track.box
                    # MOT uses 1-based frame and image coordinates; internal coordinates are 0-based.
                    mot_rows.append([frame_id + 1, track.identity, x1 + 1, y1 + 1,
                                     x2 - x1, y2 - y1, track.score, -1, -1, -1])
            elif previous is not None:
                reference = reference_interval(references, previous_time, timestamp)
                speed = interval_speed(flow, timestamp - previous_time, calibration,
                                       previous_boxes, detections[:, :4])
                row = window.update(speed, reference)
                row.update(clip_id=clip["clip_id"], frame_id=frame_id, timestamp_s=timestamp,
                           session_id=clip["session_id"], recording_id=clip["recording_id"])
                speed_rows.append(row)
            previous, previous_boxes, previous_time = rgb, detections[:, :4], timestamp
        if args.task == "tracking":
            target = output / "trackers" / "piv_flow" / "data" / f"{clip['clip_id']}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as stream:
                for row in mot_rows:
                    stream.write(",".join(str(x) for x in row) + "\n")
        else:
            write_csv(output / f"{clip['clip_id']}.csv", speed_rows)
            all_speeds.extend(speed_rows)
        print(f"Completed {clip['clip_id']}", flush=True)
    if args.task == "conveyor":
        write_csv(output / "speeds.csv", all_speeds)
        save_json(output / "metrics.json", speed_metrics(all_speeds))


if __name__ == "__main__":
    cli = parser("Fixed underground monitoring pipelines")
    cli.add_argument("--task", required=True, choices=["tracking", "conveyor"])
    cli.add_argument("--split", default="test")
    cli.add_argument("--amp", action="store_true"); cli.add_argument("--detector-only", action="store_true")
    args = cli.parse_args()
    if args.detector_only and args.task != "tracking":
        cli.error("Detector-only is a tracking diagnostic.")
    if not args.detector_only and not args.checkpoint:
        cli.error("--checkpoint is required.")
    main(args)
