import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.datasets import PairDataset
from core.datasets.io import write_flow
from core.utils.checkpoint import load_model, save_json, digest
from core.utils.config import parser, load_config


@torch.no_grad()
def evaluate(model, dataset, device, output, amp=False, save_flows=False, stress=None):
    model.eval()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in tqdm(DataLoader(dataset, batch_size=1, shuffle=False), desc="Flow evaluation"):
        index = sample["index"].item(); meta = dataset.rows[index]
        a, b, gt, valid = [sample[k].to(device) for k in ("image1", "image2", "flow", "valid")]
        luminance = .2126 * a[:, 0] + .7152 * a[:, 1] + .0722 * a[:, 2]
        dark = luminance < .2
        if stress == "gain":
            a, b = a * .5, b * .5
        elif stress == "noise":
            generator = torch.Generator(device=device).manual_seed(2026 + index)
            a = (a + .02 * torch.randn(a.shape, generator=generator, device=device)).clamp(0, 1)
            b = (b + .02 * torch.randn(b.shape, generator=generator, device=device)).clamp(0, 1)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16,
                            enabled=amp and device.startswith("cuda")):
            prediction = model(a, b)["final"].float()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite flow: {meta['pair_id']}; pixels are never silently excluded.")
        error = (prediction - gt).norm(dim=1)
        if not valid.any():
            raise ValueError(f"Empty reference support: {meta['pair_id']}")
        row = {k: meta.get(k, "") for k in ("pair_id", "base_pair_id", "group_id", "reference_id", "camera_family")}
        row["epe"] = error[valid].mean().item()
        row["valid_pixels"] = valid.sum().item()
        for name, mask in (("dark", dark), ("remaining", ~dark)):
            support = valid & mask
            row[f"epe_{name}"] = error[support].mean().item() if support.any() else ""
        rows.append(row)
        if save_flows:
            # Index filenames avoid collisions/path traversal in arbitrary pair identifiers.
            filename = f"{index:06d}.flo"
            write_flow(output / "flow" / filename, prediction[0].permute(1, 2, 0).cpu().numpy())
            row["prediction"] = "flow/" + filename
    with (output / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"pairs": len(rows), "epe": float(np.mean([r["epe"] for r in rows])), "stress": stress}
    save_json(output / "metrics.json", summary)
    return summary


if __name__ == "__main__":
    cli = parser("Evaluate on fixed FCDN/VBOF manifests")
    cli.add_argument("--manifest"); cli.add_argument("--root")
    cli.add_argument("--amp", action="store_true"); cli.add_argument("--save-flows", action="store_true")
    cli.add_argument("--stress", choices=["gain", "noise"])
    cli.add_argument("--iters", type=int); cli.add_argument("--policy", choices=["lower", "median", "no_zero", "full", "half"])
    args = cli.parse_args()
    if not args.checkpoint:
        cli.error("--checkpoint is required for evaluation")
    cfg = load_config(args.config)
    if args.policy:
        cfg["model"]["policy"] = args.policy
    if args.iters is not None:
        cfg["model"]["iters"] = args.iters
    manifest = args.manifest or cfg["data"]["eval_manifest"]
    data = PairDataset(args.root or cfg["data"].get("eval_root", cfg["data"]["root"]), manifest)
    model = load_model(cfg, args.checkpoint, args.device)
    result = evaluate(model, data, args.device, args.output, args.amp, args.save_flows, args.stress)
    result.update(checkpoint_sha256=digest(args.checkpoint), manifest_sha256=digest(manifest))
    save_json(Path(args.output) / "metrics.json", result)
    print(result)
