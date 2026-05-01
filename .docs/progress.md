# Implementation Progress

## Status as of 2026-04-29

### Completed — Steps 0–5 (inference pipeline)

| Step | Component | Status | Notes |
|---|---|---|----||
| 0 | `data/interim/pipeline.yaml` | ✅ Done | All sections; SAHI params added |
| 0 | `humming_bird_detection/config.py` | ✅ Done | `load_config()` via ruamel.yaml |
| 0 | `pyproject.toml` deps | ✅ Done | Added easyocr, click, sahi |
| 1 | `humming_bird_detection/data/metadata.py` | ✅ Done | EasyOCR + 4 regex fields |
| 1 | `tests/test_metadata.py` | ✅ Done | 3 tests pass |
| 2 | `humming_bird_detection/models/detector.py` | ✅ Done | Direct + SAHI tiled paths |
| 2 | `tests/test_detector.py` | ✅ Done | 3 tests pass (direct path) |
| 2 | `tests/test_detector_tiling.py` | ✅ Done | 4 tests pass (SAHI path) |
| 3 | `humming_bird_detection/data/preprocessing.py` | ✅ Done | pad-crop-resize |
| 3 | `tests/test_preprocessing.py` | ✅ Done | 3 tests pass |
| 4 | `humming_bird_detection/models/classifier.py` | ✅ Done | inference only; EfficientNetV2-S |
| 4 | `tests/test_classifier.py` | ✅ Done | 4 tests pass |
| 5 | `humming_bird_detection/workflow/pipeline.py` | ✅ Done | process_image + process_directory |
| 5 | `tests/test_pipeline.py` | ✅ Done | 3 tests pass |
| — | `tests/test_integration_real_image.py` | ✅ Done | 10 integration tests pass on real image |

**Test suite:** `20 unit tests + 10 integration tests` (as of 2026-04-29)  
**Run unit tests only:** `pytest tests/ -m "not integration"`  
**Run integration test:** `pytest tests/test_integration_real_image.py -v -s`

---

## Classifier training — run `20260430_193523` (2026-04-30)

| Item | Value |
|---|---|
| Run ID | `20260430_193523` (W&B: `lnjepxc0`) |
| Model | EfficientNetV2-S (ImageNet backbone + binary head) |
| Training data | `data/interim/train_data/train/` (1176 hummingbird + 2518 other WingSCapes crops) |
| Val data | `data/interim/train_data/val/` (294 hummingbird + 629 other) |
| Best epoch | 7 (early-stopped at epoch 12, patience=5) |
| val/loss | 0.0804 |
| val/accuracy | ~0.990 |
| val/f1 | ~0.980 |
| val/precision | ~0.950 |
| val/recall | ~0.990 |
| Weights | `models/hummingbird_classifier_20260430_193523.pt` (→ symlinked as `hummingbird_classifier.pt`) |

### Deferred

| Step | Component | Reason for deferral |
|---|---|---|
| 4 | `classifier.train_model()` | ✅ Implemented (2026-04-29); "other" class training data still being sourced |
| — | `scripts/prepare_training_data.py` | ✅ Done (2026-04-29); stratified 80/20 split, ImageFolder layout |
| — | `tests/test_classifier.py` (train_model tests) | ✅ Done (2026-04-29); 4 new tests, 8 total pass |
| 6 | `workflow/deduplication.py` | Not yet needed; design spec in INDEX.md |
| 7 | `cli.py` + packaging | Entry point `hbd` registered in pyproject.toml; body not implemented |
| 8 | Final end-to-end validation | Blocked on fine-tuned classifier weights |

---

## Real-image validation — `13_WSCT0033.JPG` (2026-04-29 / 2026-05-01)

**Image:** WingSCapes trail camera, 8704×4896 px, 4.3 MB JPEG  
**Ground truth:** 1 hummingbird (labeled in `13_WSCT0033.json`)

| Component | Result | Notes |
|---|---|---|
| Metadata OCR | ✅ All 4 fields parsed | temp=86°F, CAMERA 5, 12 FEB 2024, 04:18 PM |
| YOLO direct (640 px) | ❌ Bird missed | Detected feeder as "cup" only (CH-01, CH-02) |
| SAHI tiled (1280 px) | ✅ 1 bird found | Correct region; **27–34 s CPU** (warm SAHI + EasyOCR) |
| EfficientNet classifier (ImageNet only, 2026-04-29) | ⚠️ conf=0.446, label=0 | Near-random, expected (CH-05) |
| EfficientNet classifier (fine-tuned, 2026-05-01) | ❌ conf=0.362, label=0 (`hummingbird: False`) | **False negative** despite 99 % val accuracy; domain gap (CH-08) |

---

## Known issues / open questions

1. **Classifier domain gap (CH-08)** — fine-tuned model achieves 99 % val accuracy
   on iNaturalist crops but produces a false negative on the real camera-trap image
   (`conf=0.362`).  A second training round with domain-adaptation augmentation
   (color jitter, background patches) is required.

2. **OCR F→' misread** — temperature field regex broadened to accept quote
   characters as a proxy for `F`.  Should be validated on more images from
   this camera model (see DD-06, CH-03).

3. **SAHI tile count is high** — 56 tiles at 1280 px for a 8704×4624 px image.
   Increasing `slice_size` to 2048 reduces tile count to ~15 with manageable
   recall trade-off.  Benchmark needed.

4. **Python interpreter** — tests must be run with the `hbd` conda env (Python
   3.12).  VS Code interpreter should be set to
   `/home/rgutzen/miniforge3/envs/hbd/bin/python` (CH-06).
