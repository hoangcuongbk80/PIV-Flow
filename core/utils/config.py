import argparse
import copy
import json
from pathlib import Path


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path):
    path = Path(path)
    if path.suffix == ".json":
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("base", None)
    return merge(load_config(path.parent / parent), cfg) if parent else cfg


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", required=True)
    result.add_argument("--checkpoint")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int)
    result.add_argument("--output", default="outputs")
    return result
