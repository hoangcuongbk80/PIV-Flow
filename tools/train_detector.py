import argparse
from copy import deepcopy
from pathlib import Path
import torch


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Train the fixed YOLOv8s worker detector on a prepared YOLO dataset")
    cli.add_argument("--data", required=True); cli.add_argument("--device", default="0")
    cli.add_argument("--seed", type=int, default=17); cli.add_argument("--output", default="runs/detector")
    args = cli.parse_args()
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    selection = {"best": -1., "stale": 0}
    def select_by_map(trainer):
        key = "metrics/mAP50-95(B)"
        if key not in trainer.metrics:
            raise KeyError("Ultralytics no longer exposes expected mAP50-95 metric; audit version before training.")
        score = float(trainer.metrics[key])
        if score > selection["best"]:
            selection.update(best=score, stale=0)
            checkpoint_model = trainer.ema.ema if trainer.ema is not None else trainer.model
            target = Path(trainer.save_dir) / "weights" / "best_map50_95.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": deepcopy(checkpoint_model).half(), "train_args": vars(trainer.args),
                        "epoch": trainer.epoch, "best_fitness": score}, target)
        else:
            selection["stale"] += 1
        if selection["stale"] >= 20:
            trainer.stop = True
    model.add_callback("on_fit_epoch_end", select_by_map)
    # Explicit settings override Ultralytics' dataset-dependent defaults.
    model.train(data=args.data, device=args.device, optimizer="AdamW", lr0=1e-3,
                weight_decay=5e-4, batch=16, epochs=100, patience=0, seed=args.seed,
                imgsz=(480, 736), project=args.output, name="worker", exist_ok=False,
                val=True, pretrained=True)
