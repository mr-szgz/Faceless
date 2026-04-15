# Research Summary

- Research Date: `2026-04-15`
- Official Version: Ultralytics docs snapshot (page version not explicitly tagged on these pages; content references `YOLO26` and `ultralytics/engine/results.py`).
- Official Sources Used:
  - `https://docs.ultralytics.com/modes/predict/`
  - `https://docs.ultralytics.com/reference/engine/results/`
  - `https://docs.ultralytics.com/llms.txt` (checked by workflow, returned 404)
- Key Findings:
  - `model.predict()` supports a large runtime override surface including `conf`, `iou`, `imgsz`, `rect`, `device`, `batch`, `vid_stride`, `stream_buffer`, `agnostic_nms`, `classes`, `retina_masks`, `embed`, `compile`, and `end2end`.
  - Return type is `list[Results]` by default and a generator of `Results` when `stream=True`, which is the memory-safe path for long videos/streams.
  - `Results` wraps task-specific outputs in typed containers: `boxes`, `masks`, `probs`, `keypoints`, `obb`; helper methods include `summary()`, `to_df()`, `to_csv()`, `to_json()`, `save_txt()`, `save_crop()`, and device conversion (`cpu()/cuda()/numpy()/to()`).
  - `Boxes` expects shape `(N, 6)` or `(N, 7)` with columns `[x1, y1, x2, y2, (optional) track_id, conf, cls]`; `is_track=True` when 7-column input is used.
  - `OBB` expects shape `(N, 7)` or `(N, 8)` with `[x_center, y_center, width, height, rotation, (optional) track_id, conf, cls]`; exposes rotated and axis-aligned projections (`xywhr`, `xyxyxyxy`, `xyxy`).
  - `Masks` exposes contour coordinates via `xy` (pixel) and `xyn` (normalized), computed from `ops.masks2segments(...)` and scaled back to `orig_shape`.
  - `Results.update(...)` clips new boxes to original image bounds via `ops.clip_boxes(...)` before re-wrapping in `Boxes`.
- Compatibility Notes:
  - The docs note special behavior for end-to-end models (`YOLO26`, `YOLOv10`): `agnostic_nms` is limited to duplicate-label suppression there, and `end2end=False` re-enables traditional NMS + `iou` controls.
  - `rect=True` minimal padding applies cleanly for `batch=1`; for `batch>1`, identical image sizes are required in-batch or padding falls back to square `imgsz`.
  - If downstream code assumes tracking IDs, guard against `id is None` when inputs are non-tracking (`Boxes` 6 cols, `OBB` 7 cols).
