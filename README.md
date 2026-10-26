# Hummingbird Detection

[![Docs](https://github.com/Lindsay-Lab/Humming_Bird_Detection/actions/workflows/docs.yml/badge.svg)](https://Lindsay-Lab.github.io/Humming_Bird_Detection/)
[![Lint](https://github.com/Lindsay-Lab/Humming_Bird_Detection/actions/workflows/black.yml/badge.svg)](https://github.com/Lindsay-Lab/Humming_Bird_Detection/actions/workflows/black.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)


Automated detection and classification of hummingbirds in wildlife camera-trap images.

The pipeline processes batches of camera-trap JPEGs and outputs a structured CSV of bird observations. Each image is:
1. **Cropped** — the camera metadata strip is separated from the clean image frame.
2. **Detected** — YOLO finds bounding boxes around birds (direct full-image pass first, SAHI tiling fallback for small distant birds).
3. **Classified** — an EfficientNetV2-S model labels each letterboxed crop as hummingbird or other.
4. **Annotated** — date, time, temperature, and camera ID are extracted via OCR.

> **Classifier performance** (best checkpoint, val set): accuracy **99.35%** · F1 **98.98%** · precision **98.65%** · recall **99.32%**

## Pipeline overview

```
  Camera-trap JPEGs
  (input directory)
        │
        ▼
┌───────────────────┐
│   Crop margin     │  Split image into clean frame
│  (preprocessing)  │  and metadata strip (bottom)
└────────┬──────────┘
         │
         ├─────────────────────────────────────────┐
         │ clean image                             │ metadata strip
         ▼                                         ▼
┌──────────────────────────────────┐    ┌──────────────────────┐
│  Bird detection (detector.py)    │    │   OCR metadata       │
│                                  │    │  (metadata.py)       │
│  1. Direct full-image YOLO pass  │    │  date, time,         │
│     (fast; catches large birds)  │    │  temperature, camera │
│  2. SAHI tiling fallback         │    └──────────┬───────────┘
│     (catches small distant birds)│              │
└────────┬─────────────────────────┘              │
         │ per-bird bounding boxes                │
         ▼                                        │
┌───────────────────┐                             │
│  Crop & resize    │  letterbox to square,       │
│ (preprocessing)   │  then resize to 224×224     │
└────────┬──────────┘                             │
         ▼                                        │
┌───────────────────┐                             │
│  Classification   │  EfficientNetV2-S           │
│  (classifier.py)  │  hummingbird / other        │
│                   │  + class confidence         │
└────────┬──────────┘                             │
         │                                        │
         └─────────────────┬──────────────────────┘
                           ▼
             ┌─────────────────────────┐
             │   Combine per-bird      │
             │       records           │
             └─────────────┬───────────┘
                           │
                           ▼
             ┌──────────────────────────────────────┐
             │  reports/<input_name>_<timestamp>.csv │
             │        (one row per bird)             │
             └──────────────────────────────────────┘
                  from OCR metadata ──► filename, date, time, temperature, camera
                  from detection    ──► bird_index, detection_confidence
                  from classifier   ──► classification_confidence, hummingbird (bool)
```

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
# output defaults to reports/<folder_name>_<YYYYMMDD_HHMMSS>.csv
process_directory('path/to/images', None, build_detector(cfg), build_classifier(cfg), cfg)
"
```

## Documentation

Full documentation is available at **[Lindsay-Lab.github.io/Humming_Bird_Detection](https://Lindsay-Lab.github.io/Humming_Bird_Detection/)**.

| Section | Purpose |
|---------|---------|
| [Tutorial](https://Lindsay-Lab.github.io/Humming_Bird_Detection/tutorials/getting-started/) | Step-by-step first run |
| [How-to guides](https://Lindsay-Lab.github.io/Humming_Bird_Detection/how-to/run-pipeline/) | Task-oriented recipes |
| [Explanation](https://Lindsay-Lab.github.io/Humming_Bird_Detection/explanation/architecture/) | Architecture and design rationale |
| [Reference](https://Lindsay-Lab.github.io/Humming_Bird_Detection/reference/config/) | Config options and Python API |

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
docs/
  scripts/
    generate_doc_images.py  Generate documentation image assets
data/interim/pipeline.yaml  Runtime configuration
```
