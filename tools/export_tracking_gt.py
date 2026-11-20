"""Export human-verified clip-local tracking IDs to TrackEval MOT format."""
import argparse
import json
from pathlib import Path
from applications.common import read_csv


def export(root, clips, annotations, output, split="test"):
    root, output = Path(root), Path(output)
    rows = read_csv(annotations)
    selected = [c for c in read_csv(clips) if c["task"] == "tracking" and c["split"] == split]
    for clip in selected:
        clip_id = clip["clip_id"]
        crop = json.loads(clip.get("crop_xywh") or "null")
        if not crop:
            raise ValueError("Specify crop_xywh, even for a full-frame crop, to define evaluation dimensions.")
        ox, oy, width, height = crop
        frames = read_csv(root / clip["timestamps"])
        count = len(frames)
        target = output / clip_id; (target / "gt").mkdir(parents=True, exist_ok=True)
        seen = set()
        with (target / "gt" / "gt.txt").open("w", encoding="utf-8") as stream:
            for row in rows:
                if row["clip_id"] != clip_id:
                    continue
                frame_id, identity = int(row["frame_id"]), int(row["track_id"])
                if identity <= 0 or not 0 <= frame_id < count or (frame_id, identity) in seen:
                    raise ValueError("Invalid or duplicate frame/track identity in annotations.")
                seen.add((frame_id, identity))
                x, y, w, h = [float(row[k]) for k in ("x", "y", "width", "height")]
                left, top = max(x - ox, 0), max(y - oy, 0)
                right, bottom = min(x + w - ox, width), min(y + h - oy, height)
                if right <= left or bottom <= top:
                    continue
                ignored = str(row.get("ignore", "0")).lower() in ("1", "true")
                # Class 8 (distractor) invokes TrackEval's MOT ignore preprocessing.
                label = 8 if ignored else 1
                visibility = float(row.get("visibility") or 1.)
                stream.write(f"{frame_id + 1},{identity},{left + 1},{top + 1},{right - left},{bottom - top},1,{label},{visibility}\n")
        first, last = float(frames[0]["timestamp_s"]), float(frames[-1]["timestamp_s"])
        fps = (count - 1) / (last - first) if count > 1 and last > first else 30.
        (target / "seqinfo.ini").write_text(
            f"[Sequence]\nname={clip_id}\nimDir=img1\nframeRate={fps:.6f}\nseqLength={count}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n",
            encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "clips", "annotations", "output"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--split", default="test")
    args = cli.parse_args()
    export(args.root, args.clips, args.annotations, args.output, args.split)
