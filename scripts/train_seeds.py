"""Portable sequential seed launcher; run from the repository root."""
import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", default="configs/piv_s.yaml")
    cli.add_argument("--seeds", nargs="+", type=int, default=[17, 43, 89])
    cli.add_argument("--gpus", type=int, default=1); cli.add_argument("--output", default="runs/piv_s")
    args = cli.parse_args()
    for seed in args.seeds:
        command = [sys.executable]
        if args.gpus > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}"]
        command += ["train.py", "--config", args.config, "--seed", str(seed),
                    "--output", str(Path(args.output) / f"seed_{seed}")]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
