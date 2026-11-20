from core.utils.config import parser, load_config
from core.training import run_training


if __name__ == "__main__":
    args = parser("Train PIV-Flow on labeled FCDN pairs").parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_training(cfg, args.output, args.device, args.checkpoint)
