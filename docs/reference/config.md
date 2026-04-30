# Configuration reference

All settings live in `data/interim/pipeline.yaml`. Values marked **TUNABLE** are the first candidates for adjustment when adapting the pipeline to a new camera setup.

## `image`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `margin_px` | int | `272` | **TUNABLE.** Height (in pixels) of the camera metadata strip at the bottom of each image. Cropped before detection. Measure the strip height for your specific camera model. |
| `input_extension` | str | `".jpg"` | File extension glob used by `process_directory`. Case-insensitive (both `.jpg` and `.JPG` are always matched). |

## `detector`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | str | `"yolo11n.pt"` | **TUNABLE.** YOLO checkpoint filename. Larger variants (`yolo11s.pt`, `yolo11m.pt`) are more accurate at the cost of speed. |
| `model_dir` | str | `"models/YOLO"` | Directory containing the YOLO checkpoint, relative to project root. |
| `bird_class_id` | int | `14` | COCO class index for "bird". Should not need changing unless using a custom YOLO model. |
| `confidence_threshold` | float | `0.3` | **TUNABLE.** Minimum YOLO detection score. Lower values catch more birds (higher recall, lower precision). |
| `slice_size` | int or `null` | `1280` | **TUNABLE.** SAHI tile size in pixels. Set to `null` to disable tiling (faster but may miss small subjects). |
| `overlap_ratio` | float | `0.2` | **TUNABLE.** Fractional overlap between adjacent SAHI tiles. Larger values reduce missed detections at tile boundaries. |
| `nms_iou_threshold` | float | `0.5` | **TUNABLE.** IoU threshold for SAHI post-NMS tile merge. Lower values keep more boxes — reduce to ~0.3 if two nearby birds are incorrectly merged. |

## `classifier`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | str | `"hummingbird_classifier.pt"` | Classifier checkpoint filename. |
| `model_dir` | str | `"models"` | Directory containing the classifier checkpoint. |
| `image_size` | int | `224` | Input size fed to EfficientNetV2-S. Do not change unless you retrain with a different input resolution. |
| `hummingbird_label` | int | `1` | Numeric label assigned to positive (hummingbird) predictions. |
| `confidence_threshold` | float | `0.5` | **TUNABLE.** Minimum sigmoid probability to classify as hummingbird. |

## `training`

Used by `scripts/train_classifier.py` and overridable via CLI flags.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_model` | str | `"efficientnet_v2_s"` | **TUNABLE.** Backbone architecture. |
| `epochs` | int | `30` | **TUNABLE.** Total training epochs. |
| `batch_size` | int | `32` | **TUNABLE.** Batch size. Scale LR linearly when changing (`new_lr = old_lr * new_bs / old_bs`). |
| `learning_rate` | float | `0.0003` | **TUNABLE.** Head-phase learning rate. |
| `weight_decay` | float | `0.0001` | L2 regularisation. |
| `val_split` | float | `0.2` | Fraction of data reserved for validation (when val_dir is not supplied). |
| `early_stopping_patience` | int | `5` | Stop training if val loss does not improve for this many epochs. |

## `deduplication`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window_seconds` | int | `18` | **TUNABLE.** Records within this time window are considered duplicates of the same event. |

## `preprocessing`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pad_fraction` | float | `0.10` | **TUNABLE.** Fractional padding added around each detected bounding box before cropping the bird patch. 0.10 adds 10 % padding on each side. |
