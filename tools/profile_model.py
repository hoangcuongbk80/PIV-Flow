import platform
import time
from pathlib import Path
import numpy as np
import torch
from core.utils.config import parser, load_config
from core.utils.checkpoint import load_model, save_json


@torch.no_grad()
def main(args):
    model = load_model(load_config(args.config), args.checkpoint, args.device).eval()
    images = [torch.rand(1, 3, args.height, args.width, device=args.device) for _ in range(2)]
    cuda = args.device.startswith("cuda")
    def synchronize():
        if cuda:
            torch.cuda.synchronize()
    def run():
        with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.float16, enabled=args.amp and cuda):
            return model(*images)
    for _ in range(args.warmup):
        run()
    synchronize()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(args.pairs):
        synchronize(); start = time.perf_counter(); run(); synchronize()
        durations.append(time.perf_counter() - start)
    result = {"parameters": sum(p.numel() for p in model.parameters()), "mean_latency_ms": float(np.mean(durations) * 1000),
              "p95_latency_ms": float(np.quantile(durations, .95) * 1000),
              "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
              "device": torch.cuda.get_device_name() if cuda else "cpu", "torch": torch.__version__,
              "python": platform.python_version(), "input_hw": [args.height, args.width], "amp": args.amp,
              "checkpoint": args.checkpoint}
    save_json(Path(args.output) / "profile_model.json", result); print(result)


if __name__ == "__main__":
    cli = parser("Profile the flow estimator, excluding frame decode and application processing")
    cli.add_argument("--height", type=int, default=480); cli.add_argument("--width", type=int, default=736)
    cli.add_argument("--warmup", type=int, default=50); cli.add_argument("--pairs", type=int, default=500)
    cli.add_argument("--amp", action="store_true")
    main(cli.parse_args())
