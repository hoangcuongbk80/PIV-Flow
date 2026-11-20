"""Training graph, phase schedules and resumable distributed batch streams."""
import copy
import math
import os
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .datasets import PairDataset
from .datasets.augment import photometric_observation
from .geometry import warp
from .losses import sequence_loss, replay_loss
from .utils.checkpoint import seed_everything, unwrap, rng_state, restore_rng, atomic_save, digest, save_json


class BatchStream:
    """An epoch and batch cursor reconstruct data order with unchanged world/worker settings."""
    def __init__(self, dataset, batch_size, rank=0, world=1, workers=0, seed=17, state=None):
        self.dataset, self.batch_size = dataset, batch_size
        self.rank, self.world, self.workers, self.seed = rank, world, workers, seed
        self.epoch, self.cursor = 0, 0
        if state:
            self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.iterator = None

    def _open(self):
        sampler = DistributedSampler(self.dataset, num_replicas=self.world, rank=self.rank,
                                     shuffle=True, seed=self.seed, drop_last=True)
        sampler.set_epoch(self.epoch)
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 997 + self.rank)
        # Geometry is deterministically keyed by epoch and index in the dataset.
        self.dataset.epoch_seed = self.seed + self.epoch * 1_000_003
        loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler,
                            num_workers=self.workers, pin_memory=True, drop_last=True, generator=generator)
        if len(loader) == 0:
            raise ValueError("Dataset is too small for the global batch size.")
        self.iterator = iter(loader)
        for _ in range(self.cursor):
            next(self.iterator)

    def next(self):
        if self.iterator is None:
            self._open()
        try:
            value = next(self.iterator)
        except StopIteration:
            self.epoch += 1; self.cursor = 0; self._open()
            value = next(self.iterator)
        self.cursor += 1
        return value

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}


class TrainingGraph(nn.Module):
    def __init__(self, model, config, teacher=None):
        super().__init__()
        self.model, self.cfg = model, config
        # Teacher is intentionally not registered inside the DDP module.
        object.__setattr__(self, "teacher", teacher)

    def forward(self, batch, phase, real=None):
        cfg = self.cfg["train"]
        a, b, gt, valid = [batch[k] for k in ("image1", "image2", "flow", "valid")]
        a1, b1 = photometric_observation(a, b)
        policy = "full" if phase in ("warmup", "verifier") else self.model.policy
        grad = torch.no_grad() if phase == "verifier" else nullcontext()
        with grad:
            outputs = [self.model(x, y, policy=policy, return_aux=True) for x, y in ((a, b), (a1, b1))]
            supervised = sum(sequence_loss(o, gt, valid, cfg.get("gamma", .8), cfg.get("trial_weight", .5))
                             for o in outputs) / 2
        verification = supervised.new_zeros(())
        if phase != "warmup" and not cfg.get("disable_verifier_loss", False):
            verification = replay_loss(self.model, outputs, gt, valid, cfg.get("tau", .1),
                                       cfg.get("gamma", .8), cfg.get("replay", True))
        loss = (supervised if phase != "verifier" else supervised.detach()) + cfg.get("verifier_weight", 1.) * verification
        consistency = loss.new_zeros(())
        if real is not None:
            ra, rb = real["image1"], real["image2"]
            with torch.no_grad():
                forward = self.teacher(ra, rb)["final"].float()
                backward = self.teacher(rb, ra)["final"].float()
                reverse, inside = warp(backward, forward)
                finite = torch.isfinite(forward).all(1, keepdim=True) & torch.isfinite(reverse).all(1, keepdim=True)
                mask = inside & finite & ((forward + reverse).square().sum(1, keepdim=True)
                    <= .01 * (forward.square().sum(1, keepdim=True) + reverse.square().sum(1, keepdim=True)) + .5)
                forward = torch.where(mask, forward, torch.zeros_like(forward))
            sa, sb = photometric_observation(ra, rb)
            student = self.model(sa, sb)
            consistency = sequence_loss(student, forward.detach(), mask.detach(),
                                        cfg.get("gamma", .8), cfg.get("trial_weight", .5))
            loss = loss + cfg.get("consistency_weight", .1) * consistency
        return loss, {"flow_loss": supervised.detach(), "verifier_loss": verification.detach(),
                      "consistency_loss": consistency.detach()}


def phase_at(step, config, mixed=False):
    if mixed:
        return "joint", step, config.get("mixed_steps", 30_000)
    warm, initialize, joint = config.get("phases", [40_000, 20_000, 240_000])
    if step < warm:
        return "warmup", step, warm
    if step < warm + initialize:
        return "verifier", step - warm, initialize
    return "joint", step - warm - initialize, joint


def phase_lr(index, length, peak):
    warm = max(1, round(length * .05))
    if index < warm:
        return peak * (index + 1) / warm
    return peak * max(0., (length - index) / max(1, length - warm))


@torch.no_grad()
def validation_epe(model, dataset, device):
    mode = model.training; model.eval()
    errors = []
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        a, b, gt, valid = [sample[k].to(device) for k in ("image1", "image2", "flow", "valid")]
        prediction = model(a, b)["final"]
        values = (prediction - gt).norm(dim=1)[valid]
        if values.numel():
            errors.append(values.mean().item())
    model.train(mode)
    return sum(errors) / len(errors) if errors else float("inf")


def run_training(config, output, device="cuda", checkpoint=None, mixed=False):
    from .utils.checkpoint import load_model
    rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1
    if device.startswith("cuda"):
        torch.cuda.set_device(local); device = f"cuda:{local}"
    if distributed:
        dist.init_process_group("nccl" if device.startswith("cuda") and os.name != "nt" else "gloo")
    cfg = config["train"]; seed = config.get("seed", 17)
    seed_everything(seed)
    initialization_config = copy.deepcopy(config)
    if checkpoint:
        initialization_config["model"]["pretrained"] = False
    model = load_model(initialization_config, device=device)
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    if mixed and not checkpoint:
        raise ValueError("Mixed-U requires --checkpoint with an FCDN-only checkpoint.")
    resume = state is not None and state.get("mixed", False) == mixed and "optimizer" in state
    teacher = copy.deepcopy(model).eval().requires_grad_(False) if mixed else None
    if resume and mixed and "teacher" in state:
        teacher.load_state_dict(state["teacher"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2.5e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    use_amp = cfg.get("amp", True) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best = 0, float("inf")
    if resume:
        if state.get("world_size", world) != world:
            raise ValueError("Resume with the same world size; use a weights-only checkpoint to change it.")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, best = state["step"], state.get("best_epe", best)
    batch_size = cfg.get("batch_size", 8)
    if batch_size % world:
        raise ValueError("Global labeled batch size must be divisible by WORLD_SIZE.")
    dataset = PairDataset(config["data"]["root"], config["data"]["train_manifest"], True, tuple(cfg.get("crop", [368, 496])))
    dev = PairDataset(config["data"]["root"], config["data"]["dev_manifest"])
    stream_states = state.get("streams", []) if resume else []
    local_states = stream_states[rank] if stream_states else {}
    stream = BatchStream(dataset, batch_size // world, rank, world, cfg.get("workers", 4), seed,
                         local_states.get("labeled"))
    real_stream = None
    if mixed:
        real_data = PairDataset(config["data"]["real_root"], config["data"]["real_manifest"],
                                True, tuple(cfg.get("crop", [368, 496])), labeled=False)
        real_batch = cfg.get("real_batch_size", 4)
        if real_batch % world:
            raise ValueError("Global real batch size must be divisible by WORLD_SIZE.")
        real_stream = BatchStream(real_data, real_batch // world, rank, world, cfg.get("workers", 4),
                                  seed + 100, local_states.get("real"))
    if resume:
        restore_rng(state["rng_by_rank"][rank])
    else:
        seed_everything(seed + rank)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        save_json(output / "config.json", config)
        save_json(output / "data_hashes.json", {key: digest(value) for key, value in config["data"].items()
                                               if key.endswith("manifest") and value})
    graph, last_phase = None, None
    total_steps = cfg.get("mixed_steps", 30_000) if mixed else sum(cfg.get("phases", [40_000, 20_000, 240_000]))
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "events"))
    def move(batch):
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    while step < total_steps:
        phase, phase_index, phase_length = phase_at(step, cfg, mixed)
        if phase != last_phase:
            if graph is not None:
                del graph
            model.set_phase(phase)
            module = TrainingGraph(model, config, teacher)
            graph = DistributedDataParallel(module, device_ids=[local] if device.startswith("cuda") else None,
                                           broadcast_buffers=False, find_unused_parameters=True) if (
                                               distributed and any(p.requires_grad for p in model.parameters())) else module
            last_phase = phase
        model.train()
        lr = phase_lr(phase_index, phase_length, cfg.get("lr", 2.5e-4))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch, real = move(stream.next()), move(real_stream.next()) if real_stream else None
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            loss, metrics = graph(batch, phase, real)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}.")
        previous_scale = scaler.get_scale()
        updated = False
        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip_grad", 1.))
            scaler.step(optimizer); scaler.update()
            updated = scaler.get_scale() >= previous_scale
        if teacher is not None and updated:
            with torch.no_grad():
                for target, source in zip(teacher.parameters(), model.parameters()):
                    target.lerp_(source, .001)
                for target, source in zip(teacher.buffers(), model.buffers()):
                    target.copy_(source)
        step += 1
        if rank == 0 and step % cfg.get("log_every", 100) == 0:
            print(f"step={step}/{total_steps} phase={phase} loss={loss.item():.5f} lr={lr:.3g}", flush=True)
            writer.add_scalar("train/loss", loss.item(), step)
            for key, value in metrics.items():
                writer.add_scalar("train/" + key, value.item(), step)
        evaluate = step % cfg.get("validate_every", 5000) == 0 or step == total_steps
        save = evaluate or step % cfg.get("save_every", 5000) == 0
        improved = False
        if evaluate:
            if distributed:
                dist.barrier()
            if rank == 0:
                score = validation_epe(model, dev, device)
                # Select enabled-verification checkpoints for PIV-Flow, full-step for RGB.
                eligible = phase == "joint"
                improved = eligible and score < best
                if improved:
                    best = score
                writer.add_scalar("validation/epe", score, step)
                print(f"internal-development EPE={score:.6f}", flush=True)
            if distributed:
                dist.barrier()
        if save:
            local_stream = {"labeled": stream.state_dict()}
            if real_stream:
                local_stream["real"] = real_stream.state_dict()
            rngs, streams = [None] * world, [None] * world
            if distributed:
                dist.all_gather_object(rngs, rng_state()); dist.all_gather_object(streams, local_stream)
            else:
                rngs, streams = [rng_state()], [local_stream]
            if rank == 0:
                saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                         "step": step, "phase": phase, "best_epe": best, "rng_by_rank": rngs, "streams": streams,
                         "config": config, "mixed": mixed, "world_size": world}
                if teacher:
                    saved["teacher"] = teacher.state_dict()
                atomic_save(saved, output / "last.pth")
                if improved:
                    atomic_save(saved, output / "best.pth")
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()
