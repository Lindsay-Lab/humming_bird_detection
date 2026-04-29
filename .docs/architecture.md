# Pipeline Architecture

## Overview

The hummingbird detection pipeline processes wildlife-camera JPEG photographs
and produces a structured CSV of bird observation records.  Each image goes
through four sequential stages: metadata extraction, bird detection, crop
classification, and result persistence.

```
JPEG on disk
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Margin strip extraction          │
│    extract_margin(image, 272 px)    │
│    → bottom 272 rows (camera HUD)   │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐    ┌─────────────────────────────┐
    │ 2. OCR + regex      │    │ 3. SAHI tiled detection      │
    │    parse_metadata() │    │    BirdDetector.detect()     │
    │    EasyOCR (CPU)    │    │    YOLO11n × N tiles         │
    │    4 regex fields   │    │    bird class 14 filter      │
    └──────────┬──────────┘    └──────────────┬──────────────┘
               │                              │ list[Detection]
               │              ┌───────────────▼──────────────┐
               │              │ 4. Crop & resize              │
               │              │    crop_and_resize()          │
               │              │    pad 10 %, → 224×224        │
               │              └───────────────┬──────────────┘
               │                              │
               │              ┌───────────────▼──────────────┐
               │              │ 5. Binary classification      │
               │              │    HummingbirdClassifier      │
               │              │    EfficientNetV2-S head      │
               │              │    sigmoid → {0,1}            │
               │              └───────────────┬──────────────┘
               │                              │
    ┌──────────▼──────────────────────────────▼──────────────┐
    │ 6. Record assembly  process_image() → dict              │
    │    filename / date / time / camera / n_birds /          │
    │    classification_confidence / hummingbird              │
    └──────────────────────────┬─────────────────────────────┘
                               │ (skip if no detections)
                               ▼
                       observations.csv
```

## Module responsibilities

| Module | Location | Responsibility |
|---|---|---|
| `config.py` | `humming_bird_detection/config.py` | Load `pipeline.yaml` via ruamel.yaml |
| `metadata.py` | `humming_bird_detection/data/` | EasyOCR margin extraction + regex parsing |
| `preprocessing.py` | `humming_bird_detection/data/` | Pad-crop-resize around a YOLO bbox |
| `detector.py` | `humming_bird_detection/models/` | YOLO + SAHI tiling wrapper |
| `classifier.py` | `humming_bird_detection/models/` | EfficientNetV2-S binary classifier |
| `pipeline.py` | `humming_bird_detection/workflow/` | Orchestration + CSV output |

## Configuration

All tunable values live in `data/interim/pipeline.yaml`.  Parameters marked
`# TUNABLE` are the primary knobs for performance vs. accuracy trade-offs.

```yaml
image:
  margin_px: 272          # height of WingSCapes camera metadata strip

detector:
  model: "yolo11n.pt"     # TUNABLE: larger YOLO variant for more accuracy
  bird_class_id: 14       # COCO class index — "bird"
  confidence_threshold: 0.3
  slice_size: 1280        # TUNABLE: SAHI tile size; null disables tiling
  overlap_ratio: 0.2      # TUNABLE: tile overlap to avoid edge misses

classifier:
  model: "hummingbird_classifier.pt"
  confidence_threshold: 0.5
```

## Hardware assumptions

- Inference path: CPU-only (no CUDA required).
- SAHI tiling at `slice_size=1280` on an 8704×4896 image generates ≈ 56 tiles;
  runtime is ~90 s on a single CPU core.
- GPU reduces this to ~5–10 s per image.
- Training (Step 4 `train_model`, deferred) requires a GPU.
