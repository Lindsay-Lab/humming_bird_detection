# Hummingbird Detection

Automated detection and classification of hummingbirds in wildlife camera images.

The pipeline processes batches of camera-trap JPEGs and outputs a structured CSV of bird observations. Each image is:
1. **Cropped** to remove the camera metadata strip at the bottom.
2. **Detected** — YOLO finds bounding boxes around birds.
3. **Classified** — an EfficientNetV2-S model labels each crop as hummingbird or other.
4. **Annotated** — date, time, temperature, and camera ID are extracted via OCR.

## Quick start

```bash
conda env create -f environment.yml
conda activate humming_bird_detection
pip install -e .

python -c "
from humming_bird_detection.config import load_config
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.models.classifier import build_classifier
from humming_bird_detection.workflow.pipeline import process_directory

cfg = load_config()
process_directory('path/to/images', 'output.csv', build_detector(cfg), build_classifier(cfg), cfg)
"
```

## Documentation

Full documentation lives in [`docs/`](docs/index.md).

| Section | Purpose |
|---------|---------|
| [Tutorial](docs/tutorials/getting-started.md) | Step-by-step first run |
| [How-to guides](docs/how-to/) | Task-oriented recipes |
| [Explanation](docs/explanation/) | Architecture and design rationale |
| [Reference](docs/reference/) | Config options and Python API |

## Project layout

```
humming_bird_detection/   Python package
  config.py               Load pipeline.yaml
  data/
    metadata.py           OCR metadata extraction
    preprocessing.py      Crop, pad, resize
  models/
    detector.py           YOLO bird detection
    classifier.py         EfficientNetV2-S classifier
  workflow/
    pipeline.py           Orchestration
scripts/
  prepare_training_data.py  Build train/val image sets
  train_classifier.py       Fine-tune the classifier
data/interim/pipeline.yaml  Runtime configuration
```
