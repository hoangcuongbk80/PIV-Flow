# Underground applications

Both applications consume full-resolution flow and the same fixed worker detector.
No verifier confidence is used downstream. The default accuracy path processes
every input frame. See [data formats](data.md) before running.

## Registry audit and worker detector

```bash
python -m tools.audit_underground --root datasets/underground --clips datasets/underground/clips.csv
python -m tools.prepare_detector_lists --root datasets/underground --clips datasets/underground/clips.csv --output outputs/detector_data
python -m tools.train_detector --data outputs/detector_data/data.yaml --device 0 --output runs/detector
```

The split-list tool selects every third training frame within each clip and keeps
all validation/test frames. Native YOLO `labels/` stay untouched. Make sure the
box/image coordinates match; the detector's own training folders can contain the
original images, while application inference uses the configured fixed crops.
The callback saves `weights/best_map50_95.pt` by validation mAP50:95 and stops after
20 unimproved epochs, up to 100 epochs. Use the detector-development split for
training and validation.

Cache detections once for all flow methods:

```bash
python -m tools.cache_detections --root datasets/underground --clips datasets/underground/clips.csv --weights runs/detector/worker/weights/best_map50_95.pt --output datasets/underground/detections_test.csv
```

Set `application.detection_cache` to that path relative to the underground root,
or null for live detection. The cache explicitly includes empty frames. Profiling
always uses live detection and therefore requires `detector_weights` even if a
cache is configured for accuracy. Record the selected Ultralytics version.

## Tracking

The tracker aggregates flow within the middle 80% of a box, rejects component
MAD outliers, then uses the component median. At least 5% of the inner-box area
must remain. Failed aggregation reuses the last valid displacement, or zero for
a new track. Hungarian association has an IoU gate of 0.3. Three consecutive
detections confirm a track; unmatched confirmed predictions remain for at most
15 frames. Confirmed unmatched predictions and negative frames are scored.

```bash
python -m applications.run --config configs/underground.yaml --checkpoint runs/piv_s/seed_17/best.pth --task tracking --amp --output outputs/tracking
python -m applications.run --config configs/underground.yaml --task tracking --detector-only --output outputs/detector_only
python -m tools.export_tracking_gt --root datasets/underground --clips datasets/underground/clips.csv --annotations datasets/underground/tracks.csv --output outputs/tracking_gt
python -m tools.evaluate_tracking --gt outputs/tracking_gt --trackers outputs/tracking/trackers --tracker piv_flow --output outputs/tracking_metrics
```

Ground-truth IDs are positive and clip-local. Supply tracking IDs alongside the
YOLO detection labels. MOT export maps ignored boxes to distractor class 8 for
TrackEval preprocessing.
Coordinates and frames convert from internal 0-based to MOT 1-based conventions.

For paired acquisition intervals, place seed-specific result folders under one
`trackers/` parent (each contains `data/<clip>.txt`), then run:

```bash
python -m tools.bootstrap_tracking --gt outputs/tracking_gt --trackers outputs/all_trackers --control rgb_s17 rgb_s43 rgb_s89 --method piv_s17 piv_s43 piv_s89 --clips datasets/underground/clips.csv --output outputs/tracking_ci.json
```

This resamples sessions and recordings, recombines raw sequence HOTA components
with TrackEval and averages contrasts across supplied trained seeds. All
frame-level records remain within their recording cluster.

## Conveyor calibration

Measure planar point correspondences in crop coordinates:

```csv
image_x,image_y,belt_x_m,belt_y_m
100,100,0,0
600,100,1,0
600,400,1,0.6
100,400,0,0.6
```

Use at least four non-collinear points; the paper protocol specifies ten.
Supply `belt_polygon`, physical
`belt_direction` and optional `exclusion_polygons` in a geometry JSON.

```bash
python -m tools.calibrate_belt --points measured_points.csv --geometry belt_geometry.json --output datasets/underground/calibration/camera1.json
python -m applications.run --config configs/underground.yaml --checkpoint runs/piv_s/seed_17/best.pth --task conveyor --amp --output outputs/conveyor
```

The mapper projects both flow endpoints through the homography, divides their
physical displacement by the actual timestamp interval, then projects onto the
physical belt direction. Both endpoints must lie inside the belt polygon and
outside exclusion regions and enlarged worker boxes. At least 1% of the static
belt region must survive the speed MAD filter.

The causal median spans 15 intervals, with at least eight valid predictions.
Reference medians use all corresponding operational intervals independently of
prediction validity. Clip boundaries reset the window. `speeds.csv` records
availability times, eligibility, prediction validity, session and recording.

```bash
python -m tools.compare_conveyor outputs/rgb_conveyor/speeds.csv outputs/conveyor/speeds.csv --output outputs/conveyor_ci.json
```

MAE/RMSE/Within-0.10 use each method's valid support, while paired MAE differences
use common-valid timestamps. Coverage has a fixed reference-eligible denominator.
For several flow seeds, report seed metrics separately and acquisition intervals
conditional on those seeds.
