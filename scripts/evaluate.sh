#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --config configs/piv_s.yaml --checkpoint runs/piv_s/seed_17/best.pth --output outputs/piv_s/fcdn
python evaluate.py --config configs/piv_s.yaml --checkpoint runs/piv_s/seed_17/best.pth --root datasets/VBOF --manifest manifests/vbof_all.jsonl --output outputs/piv_s/vbof
