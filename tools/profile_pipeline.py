"""Decoded-buffer application timing, with optional independent DC power samples."""
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from core.utils.config import parser, load_config
from core.utils.checkpoint import load_model, save_json
from applications.common import read_csv, clip_frames, Detector, FlowPredictor, write_csv
from applications.tracking import WorkerTracker
from applications.conveyor import interval_speed, CausalSpeedWindow


def integrate_power(rows, start, end):
    times = np.array([float(r["unix_time_s"]) for r in rows])
    watts = np.array([float(r["power_w"]) for r in rows])
    order = np.argsort(times); times, watts = times[order], watts[order]
    if len(times) < 2 or (np.diff(times) <= 0).any() or times[0] > start or times[-1] < end:
        raise ValueError("DC power measurements must cover the complete measured interval with unique timestamps.")
    interior = (times > start) & (times < end)
    x = np.concatenate(([start], times[interior], [end]))
    y = np.interp(x, times, watts)
    return float(np.sum((y[1:] + y[:-1]) * .5 * np.diff(x)))


def main(args):
    import psutil
    cfg = load_config(args.config); app = cfg["application"]; root = Path(app["root"])
    clips = [r for r in read_csv(root / app["clips"]) if r["task"] == args.task and r["split"] == args.split]
    if not clips:
        raise ValueError("No matching application clips.")
    detector = Detector(app.get("detector_weights"), args.device)  # Deliberately live.
    predict = None if args.detector_only else FlowPredictor(load_model(cfg, args.checkpoint, args.device), args.device, args.amp)
    buffers = []
    for clip in clips:
        frames = []
        for item in clip_frames(root, clip):
            frames.append(item)
            if len(frames) >= args.buffer_frames:
                break
        if len(frames) < 2:
            raise ValueError("Profiling clips need at least two decoded frames.")
        calibration = json.loads((root / clip["calibration"]).read_text(encoding="utf-8")) if args.task == "conveyor" else None
        buffers.append((clip, frames, calibration))
        if len(buffers) >= args.buffer_clips:
            break
    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    def workload():
        while True:
            for clip, frames, calibration in buffers:
                tracker, window = WorkerTracker(), CausalSpeedWindow()
                previous = previous_boxes = previous_time = None
                for frame_id, stamp, rgb in frames:
                    synchronize(); start = time.perf_counter()
                    boxes = detector(rgb, clip["clip_id"], frame_id)
                    if previous is not None:
                        flow = predict(previous, rgb) if predict else None
                        if args.task == "tracking":
                            tracker.update(boxes, flow)
                        else:
                            value = interval_speed(flow, stamp - previous_time, calibration, previous_boxes, boxes[:, :4])
                            window.update(value, None)
                        synchronize()
                        yield time.perf_counter() - start
                    elif args.task == "tracking":
                        tracker.update(boxes)
                    previous, previous_boxes, previous_time = rgb, boxes[:, :4], stamp
    stream = workload(); warm_start = time.perf_counter()
    while time.perf_counter() - warm_start < args.warmup_seconds:
        next(stream)
    durations, rows = [], []
    start_wall, start_clock = time.time(), time.perf_counter()
    peak_ram = 0
    print("Measuring complete application updates...", flush=True)
    last_status = start_clock
    while time.perf_counter() - start_clock < args.measure_seconds:
        duration = next(stream)
        durations.append(duration)
        used = psutil.virtual_memory().used
        peak_ram = max(peak_ram, used)
        rows.append({"update": len(rows), "unix_time_s": time.time(), "latency_ms": duration * 1000, "system_ram_bytes": used})
        if time.perf_counter() - last_status > 30:
            print(f"Completed {len(rows)} updates", flush=True); last_status = time.perf_counter()
    end_wall = time.time(); elapsed = time.perf_counter() - start_clock
    energy = integrate_power(read_csv(args.power_csv), start_wall, end_wall) if args.power_csv else None
    result = {"updates": len(durations), "elapsed_s": elapsed, "throughput_updates_s": len(durations) / elapsed,
              "mean_latency_ms": float(np.mean(durations) * 1000), "p95_latency_ms": float(np.quantile(durations, .95) * 1000),
              "peak_system_ram_gb": peak_ram / 1e9, "dc_energy_j": energy,
              "energy_j_per_update": energy / len(durations) if energy is not None else None,
              "configured_power_mode": args.power_mode, "start_unix_s": start_wall, "end_unix_s": end_wall,
              "task": args.task, "amp": args.amp, "device": args.device, "platform": platform.platform(),
              "timing_boundary": "decoded RGB buffer through CPU application output; live detection; no capture/decode/display/network"}
    output = Path(args.output)
    write_csv(output / "latency.csv", rows); save_json(output / "profile_pipeline.json", result)
    print(result)


if __name__ == "__main__":
    cli = parser("Measure sustained application throughput, latency, memory and optional energy")
    cli.add_argument("--task", choices=["tracking", "conveyor"], required=True)
    cli.add_argument("--split", default="test"); cli.add_argument("--amp", action="store_true")
    cli.add_argument("--detector-only", action="store_true")
    cli.add_argument("--warmup-seconds", type=float, default=300)
    cli.add_argument("--measure-seconds", type=float, default=600)
    cli.add_argument("--buffer-frames", type=int, default=120)
    cli.add_argument("--buffer-clips", type=int, default=1)
    cli.add_argument("--power-mode", default="unspecified"); cli.add_argument("--power-csv")
    args = cli.parse_args()
    if args.measure_seconds <= 0 or args.buffer_frames < 2:
        cli.error("Measurement duration must be positive and buffer size at least two.")
    if args.detector_only and args.task != "tracking":
        cli.error("Detector-only is a tracking diagnostic.")
    if not args.detector_only and not args.checkpoint:
        cli.error("--checkpoint is required for a flow pipeline.")
    main(args)
