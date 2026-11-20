import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(state, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_model(config, checkpoint=None, device="cpu"):
    from core.piv_flow import PIVFlow
    model_config = dict(config["model"])
    if checkpoint:
        model_config["pretrained"] = False
    model = PIVFlow(model_config).to(device)
    if checkpoint:
        # Research checkpoints include optimizer/RNG Python objects; load only trusted files.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=True)
    return model


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
