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
| `n_birds` | int | Total birds detected in this image |
| `detection_confidence` | float | YOLO detection score |
| `classification_confidence` | float | Classifier sigmoid probability |
| `hummingbird` | int | `1` = hummingbird, `0` = other |

Returns `[]` if the image cannot be read or no birds are detected.

### `process_directory(input_dir, output_csv, detector, classifier, cfg) → DataFrame`

Process all `.jpg` / `.JPG` files in `input_dir`. Writes results to `output_csv` (parent directories created automatically).

**Returns:** `pandas.DataFrame` of all observation records.
