"""Generate YOLO split image lists while selecting every third training frame."""
import argparse
from pathlib import Path
from applications.common import read_csv


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True); cli.add_argument("--clips", required=True)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits = {"training": [], "validation": [], "test": []}; sessions = {}
    for clip in read_csv(args.clips):
        if clip["task"] != "tracking":
            continue
        split = clip["split"]
        if split not in splits:
            raise ValueError("Tracking split must be training, validation or test.")
        session = clip["session_id"]
        if session in sessions and sessions[session] != split:
            raise ValueError("A worker session crosses partitions.")
        sessions[session] = split
        # yolo_images must point to images/...; labels/... remain in native YOLO format.
        folder = root / clip["yolo_images"]
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        chosen = images[::3] if split == "training" else images
        splits[split].extend(str(p.resolve()) for p in chosen)
    for name, images in splits.items():
        (output / f"{name}.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    # JSON is also valid YAML and avoids an unnecessary dependency in this tool.
    import json
    (output / "data.yaml").write_text(json.dumps({"path": str(output.resolve()), "train": "training.txt",
        "val": "validation.txt", "test": "test.txt", "names": {0: "worker"}}, indent=2), encoding="utf-8")
    print({key: len(value) for key, value in splits.items()})
