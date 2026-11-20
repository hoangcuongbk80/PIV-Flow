#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m tools.make_smoke_data
python train.py --config configs/smoke.json --device cpu --output runs/smoke
python evaluate.py --config configs/smoke.json --checkpoint runs/smoke/best.pth --device cpu --output outputs/smoke
