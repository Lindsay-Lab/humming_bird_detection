# Challenges & Empirical Findings

This document records concrete challenges encountered during development and
validation, along with measured outcomes.  It is intended as raw material for
a future methods section in a paper or technical report.

---

## CH-01 — Small object loss due to YOLO downscaling

**Context:** Wildlife camera images are 8704×4896 px.  YOLO11n resizes all
inputs to 640×640 internally.

**Scale calculation:**
```
downscale_factor = 640 / 8704 ≈ 0.074
hummingbird in original: ~200×200 px
hummingbird after downscale: 200 × 0.074 ≈ 15 px
YOLO11n minimum reliable object size: ~20-32 px in its 640-px input space
```

**Observed outcome on `13_WSCT0033.JPG`:**
- Ground-truth bird bbox (from label JSON): center (6635, 3435), size 4130×2350 px.
- Direct YOLO inference: detected class 41 ("cup", conf=0.55) at (1918,13)–(4531,3126).
  The annotated hummingbird was **not detected**.
- SAHI tiling with 1280 px tiles: **1 bird detected** (class 14) in the correct region.

**Reference:** The SAHI paper (Akyon et al., 2022) reports consistent recall
improvements for objects smaller than 1% of image area.  A 200×200 px object
in an 8704×4896 px image covers 0.093% of area — well within the SAHI
improvement regime.

---

## CH-02 — Feeder misidentified as "cup"

**Context:** The WingSCapes camera is mounted facing a hummingbird feeder.

**Observed outcome:** YOLO11n consistently classifies the red tubular feeder as
class 41 ("cup") with high confidence (0.55 in the test image).  The bounding
box covers a large fraction of the frame (1918–4531 px wide).

**Implication for pipeline design:** A naive "keep all high-confidence
detections" strategy would always return a false positive on every feeder frame.
The class filter (`bird_class_id: 14`) correctly ignores this detection.

**Future note:** If fine-tuning YOLO for a single "hummingbird" class, the
feeder-as-cup tendency disappears because the training distribution teaches the
model what to look for.

---

## CH-03 — EasyOCR systematic misreads on WingSCapes metadata strip

**Camera model:** WingSCapes WingCam / WingScapes (trail camera for feeders).

**Strip height:** 272 px at the bottom of the 4896 px image (5.5% of height).

**Confirmed OCR artefacts on `13_WSCT0033.JPG`:**

| Ground truth | EasyOCR output | EasyOCR conf | Fix applied |
|---|---|---|---|
| `86 F` | `86 '` | 0.43 | Regex accepts `[F'\`"]` |
| `04:18 PM` | `04 : 18PM` | 0.74 | Regex allows spaces around `:` |
| `CAMERA 5` | `CAMERA 5` | 0.91+1.00 | Correct ✓ |
| `12 FEB 2024` | `12 FEB 2024` | 0.92+1.00 | Correct ✓ |
| `WINGSCAPES` | `WINGSCAPES"` | 0.96 | Not parsed (brand name only) |

**Root cause hypothesis:** The metadata strip is rendered in a sans-serif font
on a black background.  The `F` in temperature is displayed at small size (≈20
px tall in the 272 px strip) and EasyOCR's CRNN decoder substitutes visually
similar characters.  GPU inference would likely improve this.

---

## CH-04 — SAHI tile count and CPU inference time

**Image size:** 8704×4896 px  
**Margin strip removed:** 272 px → inference image: 8704×4624 px  
**SAHI config:** `slice_size=1280`, `overlap_ratio=0.2`  

**Tile count calculation:**
```
stride = 1280 × (1 - 0.2) = 1024 px
tiles_x = ceil(8704 / 1024) = 9
tiles_y = ceil(4624 / 1024) = 5
total tiles ≈ 9 × 5 = 45  (plus edge-overlap tiles → ≈56 in practice)
```

**Measured wall-clock time on single CPU core:**
- SAHI setup (first call, including SAHI model construction): ~15 s
- Per-image inference (warm, subsequent calls): **~34 s** (confirmed via integration test)
- EasyOCR (first call, downloads model weights once): ~60 s
- EasyOCR (warm): ~5–8 s

**Total process_image time (warm):** ~34–45 s (confirmed 2026-04-29)

**Mitigation options:**
- GPU: expected ~5–10× speedup → ~10–15 s per image total.
- Increase `slice_size` to 2048: ~15 tiles → ~25 s CPU, at some small-object recall cost.
- Fine-tune YOLO: smaller model, single-pass inference → < 5 s.

---

## CH-05 — Classifier returns low confidence without fine-tuned weights

**Observed:** On the test image crop (bird region, 224×224 px), the
EfficientNetV2-S model with ImageNet-only weights output confidence 0.446
(below the 0.5 threshold → `hummingbird: False`).

**Expected:** This is entirely correct behaviour.  ImageNet pre-training does
not include a "hummingbird" binary class.  The model fires the sigmoid output
near 0.5 (maximum uncertainty) on an input class it has never been trained to
distinguish.

**Action required:** Fine-tune on labeled WingSCapes crops (hummingbird /
other) before the classifier produces meaningful signal.  The detection step
(YOLO + SAHI) is decoupled from the classification step, so detections are
logged regardless — enabling bootstrapped labeling from detected crops.

---

## CH-07 — Domain gap between iNaturalist training images and camera-trap inference crops

**Context:** All currently available hummingbird training images (269 images in
`data/raw/Trochilidae/images/`) are iNaturalist research-grade photographs:
cleaned, centered subject crops, uniform lighting, no motion blur, no feeder or
branch context in frame.  At inference time, the classifier receives 224×224 crops
produced by YOLO+SAHI on 8704×4896 px WingSCapes trail-camera JPEGs.

**Expected characteristics of inference-time crops (from DD-03):**
- Subject at varying scales (SAHI tile size 1280 px; hummingbird may occupy 10–80 %
  of a tile depending on distance from camera).
- Possible partial occlusion by feeder or foliage.
- Motion blur in low-light dawn/dusk frames.
- Camera-trap colour cast (IR night mode or in-camera JPEG compression artefacts).

**Measured consequence:** With ImageNet-only weights, integration test on
`13_WSCT0033.JPG` produced `classification_confidence = 0.627` (threshold 0.5).
This is barely above the decision boundary and indicates the untuned backbone
cannot reliably discriminate hummingbirds in this domain.

**Mitigation strategy:**
1. Augment training images with `ColorJitter`, `RandomRotation` to partially
   simulate camera-trap variability.
2. After initial fine-tuning on iNaturalist data, evaluate on additional real
   camera-trap images.  If recall drops below 0.85, supplement with YOLO+SAHI
   negatives (other birds/false-positives extracted from trap footage without
   hummingbirds) for a second fine-tuning round.

**Uncertainty:** The target post-training confidence (≥ 0.85 on the real image)
is a heuristic based on the single available ground-truth sample.  Systematic
evaluation will require more annotated camera-trap frames.

---

## CH-06 — Python environment mismatch

**Problem:** The project repo sets `requires-python = "~=3.12"` in
`pyproject.toml` and the `hbd` conda env runs Python 3.12.  The shell default
(`base`) runs Python 3.10 and lacks torch, cv2, and ultralytics.

**Practical consequence:** Running `pytest` or any pipeline script requires
explicitly using `/home/rgutzen/miniforge3/envs/hbd/bin/python`.

**Persistent fix needed:** Configure VS Code's Python interpreter selector to
point to the `hbd` env for this workspace.  The `pytest` integration in the
test runner will then use the right interpreter automatically.
