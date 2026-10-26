# API reference

## `humming_bird_detection.config`

### `load_config(path=None)`

Load `pipeline.yaml` and return its contents as a plain `dict`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path \| None` | `None` | Path to a YAML config file. When `None`, loads `data/interim/pipeline.yaml` relative to the project root. |

**Returns:** `dict`

**Raises:** `FileNotFoundError` if the file does not exist; `KeyError` if any required section or key is missing.

---

## `humming_bird_detection.models.detector`

### `BirdDetector`

YOLO-based bird detector with optional SAHI tiled inference.

```python
BirdDetector(
    model_path,
    bird_class_id=14,
    confidence_threshold=0.3,
    slice_size=None,
    overlap_ratio=0.2,
    nms_iou_threshold=0.5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| Path` | — | Path to YOLO checkpoint (`.pt`). |
| `bird_class_id` | `int` | `14` | COCO class index to filter by. Validated at init. |
| `confidence_threshold` | `float` | `0.3` | Minimum detection score. |
| `slice_size` | `int \| None` | `None` | Enable SAHI tiling at this tile size (px). `None` = direct inference. |
| `overlap_ratio` | `float` | `0.2` | Fractional tile overlap for SAHI. |
| `nms_iou_threshold` | `float` | `0.5` | IoU threshold for SAHI post-NMS merge. |

**Raises** `ValueError` at init if `bird_class_id` is not present in the model's class list.

#### `.detect(image) → list[Detection]`

Run inference on a BGR uint8 NumPy array. Returns a (possibly empty) list of `Detection` objects.

**Raises** `RuntimeError` on GPU out-of-memory. Other inference errors are logged and return `[]`.

### `Detection`

`dataclass` with fields:

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `tuple[int,int,int,int]` | `(x1, y1, x2, y2)` bounding box in pixels |
| `confidence` | `float` | YOLO detection score |

### `build_detector(cfg) → BirdDetector`

Construct a `BirdDetector` from a config dict (as returned by `load_config()`).

---

## `humming_bird_detection.models.classifier`

### `HummingbirdClassifier`

EfficientNetV2-S binary classifier.

```python
HummingbirdClassifier(
    model_path=None,
    device="cpu",
    confidence_threshold=0.5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| Path \| None` | `None` | Path to a saved state dict. When `None`, uses ImageNet-pretrained weights only. |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, or `"mps"`. |
| `confidence_threshold` | `float` | `0.5` | Minimum sigmoid probability to predict label 1 (hummingbird). |

#### `.predict(images) → list[tuple[int, float]]`

Classify a list of 224 × 224 BGR uint8 NumPy arrays.

**Returns:** list of `(label, probability)` tuples where `label` is `1` (hummingbird) or `0` (other).

#### `.train_model(train_dir, val_dir, ...) → dict`

Fine-tune the classifier. See [Fine-tune the classifier](../how-to/fine-tune-classifier.md) for usage.

**Returns:** `dict` with keys `best_val_loss`, `best_epoch`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`.

### `build_classifier(cfg) → HummingbirdClassifier`

Construct a `HummingbirdClassifier` from a config dict.

---

## `humming_bird_detection.workflow.pipeline`

### `process_image(image_path, detector, classifier, cfg) → list[dict]`

Process a single image end-to-end.

**Returns:** list of observation records (one per detected bird). Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `filename` | str | Image filename |
| `date` | str \| None | Date from OCR metadata |
| `time` | str \| None | Time from OCR metadata |
| `camera` | str \| None | Camera ID from OCR metadata |
| `bird_index` | int | Index of this bird within the image |
| `detection_confidence` | float | YOLO detection score |
| `classification_confidence` | float | Classifier sigmoid probability |
| `hummingbird` | int | `1` = hummingbird, `0` = other |
| `bbox_x1` | int | Bounding box x1 in original-image pixels |
| `bbox_y1` | int | Bounding box y1 in original-image pixels |
| `bbox_x2` | int | Bounding box x2 in original-image pixels |
| `bbox_y2` | int | Bounding box y2 in original-image pixels |

Returns `[]` if the image cannot be read or no birds are detected.

### `process_directory(input_dir, output_csv, detector, classifier, cfg) → DataFrame`

Process all `.jpg` / `.JPG` files in `input_dir`. Writes results to `output_csv` (parent directories created automatically).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | `str \| Path` | — | Directory containing JPEG images to process. |
| `output_csv` | `str \| Path \| None` | `None` | Output path for the CSV. When `None`, auto-generates a timestamped path under `reports/`. |
| `detector` | `BirdDetector` | — | Configured detector instance. |
| `classifier` | `HummingbirdClassifier` | — | Configured classifier instance. |
| `cfg` | `dict` | — | Config dict as returned by `load_config()`. |

**Returns:** `pandas.DataFrame` of all observation records (one row per detected bird across all images).  The DataFrame includes post-processed columns (`is_duplicate`, `is_suspected_duplicate`) when the corresponding config sections are present.  A companion `_per_photo.csv` is written alongside the bird-level CSV.  Total image and blank-image counts are stored in `df.attrs` (`total_images`, `blank_images`).

---

## `humming_bird_detection.workflow.postprocessing`

### `flag_duplicates(df, window_seconds) → DataFrame`

Add an `is_duplicate` column flagging motion-event duplicates within `window_seconds` of the most recent non-duplicate image.  Successive images are ordered by (parsed_datetime, filename).  Rows with unparseable timestamps are treated as unique.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | Bird-level DataFrame from `process_directory`. |
| `window_seconds` | `float` | Maximum gap in seconds for the later image to be considered a duplicate. |

**Returns:** Copy of `df` with an added `is_duplicate` boolean column.

### `flag_suspected_duplicates(df, window_seconds=60) → DataFrame`

Add an `is_suspected_duplicate` column using composition-based duplicate detection.  Within each (camera, date, minute-bucket) group, photos with identical bird-composition signatures — (num_hummingbirds, num_other) — are clustered.  Among *N* photos with identical composition, only `ceil(N / 3)` are kept; the remainder are flagged.  The rule encodes:

| Photos (N) | Kept | Flagged |
|------------|------|---------|
| 1 | 1 | 0 |
| 2–3 | 1 | 1–2 |
| 4–6 | 2 | 2–4 |
| 7–9 | 3 | 4–6 |

All bird-level rows sharing the same filename receive the same flag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `DataFrame` | — | Bird-level DataFrame. Must contain `filename`, `camera`, `date`, `time`, `is_hummingbird`. |
| `window_seconds` | `float` | `60` | Time window for grouping photos. Default 60 groups within the same clock minute. |

**Returns:** Copy of `df` with an added `is_suspected_duplicate` boolean column.

### `aggregate_to_photos(df) → DataFrame`

Aggregate bird-level records to one row per photo.  Each output row contains per-photo counts, confidence statistics, and propagated duplicate flags.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Source image filename |
| `date` | str | Date from OCR |
| `time` | str | Time from OCR |
| `temperature` | float | Temperature from OCR |
| `camera` | str | Camera ID |
| `num_total_birds` | int | Total birds detected in this photo |
| `num_hummingbirds` | int | Hummingbird count |
| `num_other` | int | Other bird count |
| `mean_detection_conf` | float | Mean YOLO confidence |
| `max_detection_conf` | float | Max YOLO confidence |
| `mean_classification_conf` | float | Mean classifier confidence |
| `max_classification_conf` | float | Max classifier confidence |
| `has_hummingbird` | bool | At least one hummingbird present |
| `is_duplicate` | bool | Propagated motion-event flag (if present) |
| `is_suspected_duplicate` | bool | Propagated suspected flag (if present) |

### `build_postprocessors(cfg) → list[Callable]`

Return a list of post-processor callables implied by the config.  Registers `flag_duplicates` when the `deduplication` section is present, and `flag_suspected_duplicates` when the `suspected_duplicates` section is present.  Processors are applied in order.

### `apply_postprocessors(df, processors) → DataFrame`

Apply a sequence of post-processors to a DataFrame in order.

---

## `humming_bird_detection.workflow.analysis`

### `compute_summary(df, total_images=None) → dict`

Return a flat dict of aggregate pipeline statistics including total/blank images, bird counts, confidence means/medians, duplicate rates, and suspected-duplicate rates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `DataFrame` | — | Bird-level DataFrame from `process_directory`. |
| `total_images` | `int` | `None` | Total JPEGs processed. When `None`, read from `df.attrs["total_images"]` or estimated from unique filenames. |

**Returns:** dict with keys `total_images`, `blank_images`, `total_detections`, `hummingbird_detections`, `other_detections`, `hummingbird_rate`, `mean_detection_conf`, `median_detection_conf`, `mean_classification_conf`, `median_classification_conf`, `mean_classifier_certainty`, `median_classifier_certainty`, `mean_hummingbird_conf`, `mean_other_conf`, plus `duplicate_detections`/`duplicate_rate` and `suspected_duplicate_detections`/`suspected_duplicate_rate` when the corresponding columns are present.

!!! note "Classifier certainty"
    `classification_confidence` is the sigmoid probability of the *hummingbird* class (P(hummingbird)). For birds classified as "other" this value is *low* (e.g., 0.0001 = very certain it is NOT a hummingbird). The summary therefore provides:
    
    - **`classifier_certainty`** — confidence in the *predicted* class (p for hummingbirds, 1−p for others; 1.0 = fully certain).
    - **`mean_hummingbird_conf`** / **`mean_other_conf`** — per-class raw probability means, interpretable separately.

### `compute_per_camera(df) → DataFrame`

Return a per-camera breakdown with detection counts, hummingbird rate, mean confidences, blank-image counts, and efficiency (photos with birds / total photos per camera).

### `compute_suspected_duplicate_summary(df) → dict`

Return statistics about composition-based suspected duplicates at both photo and bird-record granularity.

### `compute_hourly_activity(df) → DataFrame`

Return detections binned by hour of day (0–23) with columns `hour`, `total`, `hummingbird`, `other`.

### `compute_daily_activity(df) → DataFrame`

Return detections aggregated by calendar date.

### `compute_per_camera_hourly(df) → dict[str, DataFrame]`

Return a dict mapping camera name to its hourly-activity DataFrame.

### `compute_confidence_correlation(df) → dict`

Return Pearson correlation (`pearson_r`, `p_value`) between detection confidence and **classifier certainty** (confidence in the predicted class). Certainty is used instead of raw `classification_confidence` because the raw value is inversely coded for "other" records — correlating it with detection confidence across mixed classes mostly measures class balance, not any detector↔classifier relationship.

### `generate_report(df, cfg=None) → str`

Produce a human-readable terminal report covering overview stats, duplicate analysis, per-camera breakdowns with efficiency, and hourly/daily activity bar charts.

### `export_summary_json(df, output_path, cfg=None) → Path`

Write a machine-readable JSON summary to `output_path` including aggregate stats, per-camera breakdown, suspected-duplicate summary, hourly/daily activity, confidence correlation, and config metadata.

### `generate_plots(df, output_dir, cfg=None) → list[Path]`

Generate publication-ready PNG plots: confidence histograms, hourly activity stacked bars, per-camera grouped bars, and daily timeline.  Returns absolute paths to saved files.
