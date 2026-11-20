"""Use python -m tools.diagnose; a frozen RGB bank supports common-candidate tests."""
import csv
from pathlib import Path
import torch
from core.datasets import PairDataset
from core.utils.config import parser, load_config
from core.utils.checkpoint import load_model
from core.datasets.augment import photometric_observation
from evaluation.diagnostics import diagnose


@torch.no_grad()
def main(args):
    cfg = load_config(args.config)
    model = load_model(cfg, args.checkpoint, args.device).eval()
    bank = load_model(load_config(args.bank_config), args.bank_checkpoint, args.device).eval() if args.bank_checkpoint else model
    if args.bank_checkpoint:
        bank.policy = "full"
    dataset = PairDataset(args.root or cfg["data"].get("eval_root", cfg["data"]["root"]), args.manifest or cfg["data"]["eval_manifest"])
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]; meta = dataset.rows[index]
        a, b, gt, valid = [sample[k][None].to(args.device) for k in ("image1", "image2", "flow", "valid")]
        torch.manual_seed(2026 + index)
        a1, b1 = photometric_observation(a, b)
        observations = [(a, b), (a1, b1)] if args.bank_checkpoint else [(a, b)]
        bank_outputs = [bank(x, y, return_aux=True) for x, y in observations]
        # Verifier evidence belongs to the evaluated model, not the bank's learned encoder.
        evidence_outputs = [model(x, y, return_aux=True) for x, y in observations]
        for origin, output in enumerate(bank_outputs):
            for iteration, trace in enumerate(output["traces"]):
                location_mask = None
                if args.locations:
                    shape = trace["anchor"].shape
                    generator = torch.Generator(device=args.device).manual_seed(2026 + index * 100 + origin * 10 + iteration)
                    chosen = torch.randperm(shape[-2] * shape[-1], generator=generator, device=args.device)[:args.locations]
                    location_mask = torch.zeros((1, 1, shape[-2] * shape[-1]), device=args.device, dtype=torch.bool)
                    location_mask[:, :, chosen] = True
                    location_mask = location_mask.reshape(1, 1, *shape[-2:])
                for observation, evaluated in enumerate(evidence_outputs):
                    row = diagnose(model, evaluated["evidence"], trace, gt, valid, location_mask, args.policy)
                    if row:
                        rows.append(dict(pair_id=meta["pair_id"], iteration=iteration + 1,
                                         origin=origin, observation=observation, **row))
    path = Path(args.output) / "diagnostics.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No nondegenerate, reference-valid comparisons.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    cli = parser("Measure own-trajectory or common-candidate verification")
    cli.add_argument("--root"); cli.add_argument("--manifest")
    cli.add_argument("--bank-config"); cli.add_argument("--bank-checkpoint")
    cli.add_argument("--locations", type=int, default=20)
    cli.add_argument("--policy", default="lower", choices=["lower", "median", "no_zero"])
    args = cli.parse_args()
    if not args.checkpoint or (args.bank_checkpoint and not args.bank_config):
        cli.error("Provide --checkpoint; a bank also requires --bank-config.")
    main(args)
