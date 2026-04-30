# Architecture

The pipeline transforms a raw camera-trap image into a structured record that answers: *is there a hummingbird in this photo, and if so where?*

## Pipeline stages

```
Raw image
    │
    ▼
┌─────────────────────┐
│  OCR metadata strip │  easyocr reads temperature, date/time from the bottom strip
└────────┬────────────┘
         │ margin_px crop removed before detection
         ▼
┌─────────────────────┐
│   Bird detection    │  YOLO11n, COCO class 14 ("bird")
│   (+ SAHI tiling)  │  optional sliding-window tiling for small subjects
└────────┬────────────┘
         │ one bounding-box crop per detected bird
         ▼
┌─────────────────────┐
│  Hummingbird        │  EfficientNetV2-S binary classifier
│  classification     │  outputs probability ∈ [0, 1]
└────────┬────────────┘
         │
         ▼
Structured record (CSV row)
  filename, timestamp, temperature,
  bbox, confidence, is_hummingbird
```

## The camera-trap metadata strip

Camera-trap images often contain a burned-in status strip at the bottom (temperature, date/time, moon phase, sequence number). The `margin_px` setting tells the pipeline where this strip starts so it is cropped away before detection — otherwise the OCR digits can trigger false bird detections.

## SAHI tiling

Standard YOLO inference resizes the whole image to 640 × 640 before inference. Hummingbirds are small subjects and can be missed at this resolution. SAHI (Slicing Aided Hyper Inference) divides the image into overlapping tiles, runs YOLO on each, then merges the detections with NMS. This recovers small-subject detections at the cost of higher inference time.

Tiling is controlled by the `detector.use_tiling` flag in `pipeline.yaml`.

## Two-stage architecture

A two-stage design (detect then classify) is more robust than a single-stage hummingbird detector for this dataset because:

- YOLO11n pre-trained on COCO already detects birds reliably with no fine-tuning.
- The binary classifier can be re-trained quickly on a small, domain-specific dataset without affecting detection quality.
- Separating the stages makes evaluation and debugging easier — detection failures and classification failures have different diagnostics.

## Two-phase fine-tuning

The classifier is fine-tuned in two phases:

| Phase | Epochs | Weights updated | Learning rate |
|-------|--------|----------------|---------------|
| 1 (head) | 1–5 | Classification head only | `lr` |
| 2 (full) | 6+ | All weights | `lr/10`, cosine annealed to 0 |

Freezing the backbone in phase 1 prevents catastrophic forgetting of ImageNet features while the new head converges. Full fine-tuning in phase 2 adapts low-level features to camera-trap image characteristics.
