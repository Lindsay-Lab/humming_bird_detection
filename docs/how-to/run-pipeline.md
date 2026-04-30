# Run the pipeline on a directory of images

## What you need

- Package installed (`pip install -e .`)
- YOLO checkpoint at `models/YOLO/yolo11n.pt`
- Classifier checkpoint at `models/hummingbird_classifier.pt`
- `data/interim/pipeline.yaml` present (copy from the example in the repo root)

## Steps

### 1. Load config and build components

```python
from humming_bird_detection.config import load_config
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.models.classifier import build_classifier

cfg        = load_config()                  # reads data/interim/pipeline.yaml
detector   = build_detector(cfg)
classifier = build_classifier(cfg)
```

To use a different config file:

```python
cfg = load_config("path/to/my_config.yaml")
```

### 2. Process a directory

```python
from humming_bird_detection.workflow.pipeline import process_directory

df = process_directory(
    input_dir="data/raw/station_A",
    output_csv="data/processed/station_A_observations.csv",
    detector=detector,
    classifier=classifier,
    cfg=cfg,
)
```

`process_directory` processes all `.jpg` / `.JPG` files in `input_dir`, writes the CSV, and returns the DataFrame.

### 3. Process a single image

```python
from humming_bird_detection.workflow.pipeline import process_image

records = process_image("data/raw/station_A/IMG_0042.JPG", detector, classifier, cfg)
# records is a list of dicts (one per detected bird, empty if no birds)
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| No detections | `confidence_threshold` too high | Lower `detector.confidence_threshold` in config |
| Many false positives | Threshold too low | Raise `detector.confidence_threshold` |
| OCR fields all `None` | Wrong `margin_px` | Measure your camera strip height and update `image.margin_px` |
| `FileNotFoundError` on config | Missing `pipeline.yaml` | Copy `data/interim/pipeline.yaml` from the repo |
