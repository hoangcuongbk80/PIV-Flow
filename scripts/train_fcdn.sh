#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc_per_node=2 train.py --config configs/piv_s.yaml --seed 17 --output runs/piv_s/seed_17
