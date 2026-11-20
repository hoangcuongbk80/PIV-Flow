# Experiments and outputs

## Configurations

| Config | Experiment |
|---|---|
| `piv_s.yaml`, `piv_m.yaml` | Full FCDN-only models |
| `rgb_s.yaml`, `rgb_m.yaml` | Same proposal model, full-step recurrence |
| `mean_gain.yaml`, `marginal_quantile.yaml` | Full-schedule matched utility controls |
| `short.yaml` | Complete short schedule: 40k + 20k + 60k |
| `independent_augmentation.yaml` | No cross-observation replay |
| `no_endpoint.yaml`, `no_trial.yaml` | Endpoint and trial-loss ablations |
| `confidence.yaml`, `action_gate.yaml`, `mean_error.yaml` | Auxiliary gate controls |
| `half_step.yaml` | Retrained fixed half-step model |
| `parameter_matched_rgb.yaml` | RGB-S with a functional, approximately capacity-matched motion adapter |
| `mixed_u.yaml` | Canon-held-out adaptation example |
| `underground.yaml` | Applications with fixed detector |
| `smoke.json` | Small generated-data training and evaluation run |

Seeds are CLI overrides. Quantile sensitivity uses an inherited config with
`train.tau: 0.05` or `0.25`. Inference-policy sensitivity uses `--policy median` or
`no_zero` on the **same checkpoint**; iteration sensitivity uses `--iters`.
Changing inference iterations does not retrain the model.

To share the seed-matched warm-up across short branches, keep the exact 40k-step
checkpoint (`last.pth` saved at that step), then pass it to each branch's
`train.py --checkpoint ...`. The phase/optimizer/data cursor resume from step
40k. Use a distinct output folder for every branch. Full-schedule and short
configs have identical first-phase lengths. Do not seed a short branch from a
checkpoint saved after joint training.

## Acquisition and seed statistics

Per-run `pairs.csv` contains one row per pair and pair-mean EPE. To report several
seeds, first create seed-averaged pair rows and separate seed SD:

```bash
python -m tools.aggregate_runs --runs outputs/s17/pairs.csv outputs/s43/pairs.csv outputs/s89/pairs.csv --output outputs/pooled
python -m tools.compare outputs/rgb_pooled/pairs.csv outputs/piv_pooled/pairs.csv --dataset vbof
```

Bootstrap resamples FCDN base IDs or VBOF acquisition groups within camera-family
strata. It preserves all exposure rows of each selected group. These intervals
are conditional on supplied trained seeds; seed SD is not absorbed into an
acquisition interval. Individual two/three-seed SDs should not be interpreted as
precise variance estimates. Pair support must match exactly.

`tools.diagnose` exports per-pair/iteration origin/evidence records. For the
common-candidate study use a frozen RGB-S warm-up as bank and each verifier's
60k-step initialization checkpoint, with `--locations 20`. Same pair-index-based
noise draws and source locations are used for every head. Source sampling is
without replacement over the feature grid. Empty supports are omitted; aggregate
each reported rate over its nonempty pair/iteration groups. Tail fractions use
strict `target < predicted_quantile`. Own-trajectory and common-bank diagnostics
answer different questions and must be labeled separately.

## External baselines

For external baselines, run their official code and export full-resolution
HxWx2 `.flo` predictions with a CSV mapping:

```csv
pair_id,prediction
example_pair,flow/example_pair.flo
```

```bash
python -m tools.evaluate_predictions --root datasets/VBOF --manifest manifests/vbof_all.jsonl --predictions /path/to/baseline/index.csv --output outputs/baseline
```

## Profiling

```bash
python -m tools.profile_model --config configs/piv_s.yaml --checkpoint runs/piv_s/seed_17/best.pth --amp --output outputs/profile_model
python -m tools.profile_pipeline --config configs/underground.yaml --checkpoint runs/piv_s/seed_17/best.pth --task tracking --amp --power-mode 30W --output outputs/profile_tracking
```

Model profiling uses 50 warm-ups and 500 synchronized pairs. Application profiling
decodes buffers first, then includes tensor preparation/transfers, live detection,
flow, downstream processing and CPU outputs. It excludes camera acquisition,
decoding, display and network transport. Buffers replay sequential frame pairs;
they do not simulate a lower-rate schedule. Application defaults are five minutes
of warm-up and ten minutes of measurement. Repeat three independent measurement
sessions for each task/power mode and retain raw `latency.csv` files.

The `--power-mode` value is a recorded label, not a hardware reconfiguration.
Configure/check the Jetson mode using NVIDIA's documented device tools before
measurement. Peak system RAM includes the machine's other processes and decoded
buffers; report that boundary. Allocated CUDA memory is a separate model metric.

Optional independent meter CSV:

```csv
unix_time_s,power_w
1800000000.0,31.2
1800000000.1,31.4
```

`--power-csv` integrates DC-input samples covering the full measurement window and
divides by every completed update, including invalid speed outputs. Without
measurements, energy fields are null.
