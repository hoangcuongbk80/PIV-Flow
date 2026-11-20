# Data formats

## Public data provenance

Use the original [Optical Flow in the Dark project](https://github.com/mf-zhang/Optical-Flow-in-the-Dark)
for released FCDN/VBOF assets. IGEFlow documents preparation from FlyingChairs and
VBOF RAW using its `syndata.py` and `vbofdata.py`:
https://github.com/HDU-ASL/IGEFlow

Use an existing FCDN release and processed, geometrically aligned RGB for VBOF.
For RAW input, use the release's processing settings for demosaicing, white
balance, tone/gamma, orientation, normalization and output dimensions.

Prepare the VBOF index with acquisition metadata, including capture IDs, shared
reference IDs and acquisition groups. Underground clip registries include session,
recording and reference metadata as described below.

## FCDN index CSV

Required columns:

```csv
pair_id,base_pair_id,image1,image2,flow,official_split
chair_00001_dark1,00001,data/00001_img1.png,data/00001_img2.png,data/00001_flow.flo,training
```

- All paths are relative to the configured dataset root.
- `pair_id` identifies the released observation pair; `base_pair_id` identifies
  the original motion/geometry across photometric variants.
- `official_split` is `training` or `validation`, obtained from the official
  release's split, not randomly inferred.
- Optional `valid_mask` is an image whose nonzero pixels mark valid references.
- The internal split sorts unique training base IDs and uses NumPy PCG64 seed
  2026; the first 2,224 permuted IDs form development. All variants stay together.
- `--paper-protocol` checks for 20,008 optimization, 2,224 development and 640
  validation observation pairs.

For a conventional release layout, `prepare_fcdn --official-split ...` recognizes
`*_img1.{png,jpg,ppm}`, `*_img2.*`, and `*_flow.flo` (or matching `*.flo`). It checks
unique pairing and assumes split rows correspond to sorted unique base IDs.
Use an explicit CSV whenever that assumption is not true.

## VBOF index CSV

```csv
pair_id,image1,image2,flow,valid_mask,reference_id,capture_id,group_id,camera_subset,camera_family,exposure1_s,exposure2_s
sony1_pair_001,sony1/a.png,sony1/b.png,sony1/reference.flo,,sony1_r001,sony1_capture001,sony1_group001,sony1,Sony,0.01,0.01
```

`reference_id`, `capture_id` and `group_id` must be globally scoped identifiers.
Alternative exposures of one capture and pairs sharing a directed reference must
belong to one acquisition group. `camera_family` is one of `Canon`, `Fuji`,
`Nikon`, `Sony`; `camera_subset` retains finer camera identity. Exposure metadata
is optional and is never inferred from RGB brightness.

Forward references are expressed in input-image pixels on image1's grid.
Supported files: Middlebury `.flo`, `.pfm` (first two channels), `.npy` HxWx2, and
`.npz` with a `flow` array. Missing values need explicit validity masks or
nonfinite reference values. No default magnitude cutoff changes evaluation
support. A flow color visualization is never accepted as a numerical reference.

Prepare one all-pair manifest and four camera-held-out folds:

```bash
python -m tools.prepare_vbof --index /path/to/vbof_index.csv --output manifests
python -m tools.audit_data --root datasets/VBOF --manifests manifests/vbof_canon_unlabeled.jsonl manifests/vbof_canon_test.jsonl --pixels
```

The unlabeled adaptation manifests remove `flow` and `valid_mask` paths. Reference
IDs remain as grouping metadata only. Use `--paper-protocol` to check 2,349 pairs,
426 directed references, 214 acquisition groups and eight camera subsets.

## Underground clip registry

Private data are never used to train the optical flow model. Store one row per
clip in `clips.csv`. Example (illustrative identifiers and files only):

```csv
clip_id,task,split,session_id,recording_id,video,frames,timestamps,crop_xywh,calibration,reference,yolo_images
worker001,tracking,test,S07,R071,videos/worker001.mp4,,timestamps/worker001.csv,"[100, 100, 736, 480]",,,images/test/worker001
belt001,conveyor,test,S07,R072,videos/belt001.mp4,,timestamps/belt001.csv,"[100, 100, 736, 480]",calibration/camera1.json,references/belt001.csv,
```

- `task`: `tracking` or `conveyor`.
- Worker `split`: `training`, `validation`, `test`; belt split: `development`, `test`.
- Either `video` or `frames` supplies RGB input. Frame folders are sorted by
  filename; use zero-padded numeric filenames in temporal order.
- `crop_xywh` uses original image pixels. Crops are applied without resizing.
- Every timestamp file has `frame_id,timestamp_s`, frame IDs starting at zero.
  Timestamps use the same clock as operational-speed intervals and must increase.
- Registry paths are relative to `application.root`.
- Session/recording IDs must not cross development and test partitions.

## YOLOv8 detection and tracking IDs

Keep the existing YOLO image/label correspondence, with normalized
`class_id x_center y_center width height`. The detector uses class 0 for workers.
`yolo_images` points to each clip's folder under `images/...`; its parallel
`labels/...` folder remains in native YOLO format. Preserve empty negative frames.

For tracking evaluation supply one additional `tracks.csv`:

```csv
clip_id,frame_id,track_id,x,y,width,height,ignore,visibility
worker001,0,1,120,140,70,190,0,1.0
worker001,1,1,122,140,70,190,0,1.0
```

These boxes use **original-frame, zero-based pixel coordinates**, unlike YOLO's
normalized detection boxes. Identity is positive and clip-local, consistent over
time. `ignore` marks boxes excluded as eligible workers; optional `visibility`
defaults to one. Missing identity annotations cannot be filled by the tracker
being evaluated. `export_tracking_gt` crops/clips boxes and converts coordinates
to MOT's 1-based frame/image convention.

Detection caches instead use **crop-local zero-based xyxy** boxes:

```csv
clip_id,frame_id,x1,y1,x2,y2,score
worker001,0,20,40,90,230,0.92
worker001,1,,,,,
```

The second row explicitly declares a frame with no detections. A missing cache
frame raises an error; it is not silently interpreted as a negative frame.

## Conveyor calibration and references

Calibration geometry uses **crop-local** pixels. A camera calibration JSON has:

```json
{
  "coordinate_system": "crop_pixels_to_belt_meters",
  "homography": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 1]],
  "belt_direction": [1, 0],
  "belt_polygon": [[0, 0], [735, 0], [735, 479], [0, 479]],
  "exclusion_polygons": []
}
```

Replace the example homography with one estimated from measured points using
`tools.calibrate_belt`.
`belt_direction` is a physical-plane vector, normalized by the estimator. Its
sign defines the positive belt-speed direction. Crop coordinates must agree
between calibration, detector cache and flow.

Operational references are piecewise interval averages:

```csv
start_time_s,end_time_s,speed_mps,valid
0.000000,0.033333,0.0,1
0.033333,0.066667,0.1,1
```

Intervals must cover frame-pair times completely, without gaps or overlaps.
Code computes duration-weighted inter-frame averages, then a median of all 15
reference intervals independently of predicted validity. The first 14 output
intervals are warm-up. At least eight valid predicted intervals are needed.
Output time is availability after the second frame of the newest interval.
