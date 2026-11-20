from pathlib import Path
import cv2
import torch
from PIL import Image
from core.datasets.io import read_image, write_flow
from core.utils.checkpoint import load_model
from core.utils.config import parser, load_config
from core.utils.visualization import flow_rgb


def tensor(rgb, device):
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 255


@torch.no_grad()
def main(args):
    model = load_model(load_config(args.config), args.checkpoint, args.device).eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    def predict(a, b, name):
        with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.float16,
                            enabled=args.amp and args.device.startswith("cuda")):
            flow = model(tensor(a, args.device), tensor(b, args.device))["final"]
        flow = flow[0].permute(1, 2, 0).float().cpu().numpy()
        write_flow(output / f"{name}.flo", flow)
        Image.fromarray(flow_rgb(flow)).save(output / f"{name}.png")
    if args.image1:
        predict(read_image(args.image1), read_image(args.image2), "flow")
    elif args.frames:
        paths = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".ppm"))
        if len(paths) < 2:
            raise ValueError("Need at least two ordered frames.")
        for i, (a, b) in enumerate(zip(paths[:-1], paths[1:])):
            predict(read_image(a), read_image(b), f"{i:06d}")
    else:
        capture = cv2.VideoCapture(args.video)
        ok, previous = capture.read(); i = 0
        if not ok:
            raise ValueError("Cannot read the input video.")
        try:
            while True:
                ok, current = capture.read()
                if not ok:
                    break
                predict(cv2.cvtColor(previous, cv2.COLOR_BGR2RGB), cv2.cvtColor(current, cv2.COLOR_BGR2RGB), f"{i:06d}")
                previous = current; i += 1
        finally:
            capture.release()


if __name__ == "__main__":
    cli = parser("Visualize PIV-Flow on RGB pairs, frames or a video")
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--image1"); group.add_argument("--frames"); group.add_argument("--video")
    cli.add_argument("--image2"); cli.add_argument("--amp", action="store_true")
    args = cli.parse_args()
    if not args.checkpoint or (args.image1 and not args.image2):
        cli.error("Provide --checkpoint and --image2 when using --image1.")
    main(args)
