# Challenges & Empirical Findings

This document records concrete challenges encountered during development and
validation, along with measured outcomes.  It is intended as raw material for
a future methods section in a paper or technical report.

---

## CH-01 — Small object loss due to YOLO downscaling

**Context:** Wildlife camera images are 8704×4896 px.  YOLO11n resizes all
inputs to 640×640 internally.

**Scale calculation (typical feeder bird at normal distance):**
```
downscale_factor = 640 / 8704 ≈ 0.074
hummingbird in original: ~200–500 px (estimated; varies with distance)
hummingbird after downscale: 200 × 0.074 ≈ 15 px
YOLO11n minimum reliable object size: ~20-32 px in its 640-px input space
```

**Correction (2026-05-01):** The test image `13_WSCT0033.JPG` has an
exceptionally large bird bbox: **4130×2350 px** (bird very close to camera).
The 200 px estimate above applies to birds at normal feeder distance.  The
large-bird detection failure in the test image is explained separately in
CH-09.

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

## CH-05 — Classifier confidence without fine-tuned weights (baseline, superseded by CH-08)

**Observed (2026-04-29, ImageNet weights only):** On the test image crop (bird
region, 224×224 px), the EfficientNetV2-S model with ImageNet-only weights
output confidence 0.446 (below the 0.5 threshold → `is_hummingbird: False`).

**Expected:** Correct behaviour — ImageNet pre-training does not include a
"hummingbird" binary class.  The model fires the sigmoid output near 0.5
(maximum uncertainty) on an input class it has never been trained to
distinguish.

**Resolution:** Fine-tuning run `20260430_193523` completed (see CH-08).
ImageNet-only baseline is superseded.

---

## CH-07 — Training data is WingSCapes camera-trap crops (not iNaturalist)

**Context:** Training data in `data/interim/train_data/` consists of labeled
camera-trap crops from WingSCapes WSCT images (sourced via Roboflow, filenames
contain `.rf.` and `WSCT` identifiers).  Class counts:

| Split | hummingbird | other | total |
|---|---|---|---|
| train | 1176 | 2518 | 3694 |
| val | 294 | 629 | 923 |

The iNaturalist Trochilidae images in `data/raw/Trochilidae/` are a separate
asset and were **not** used in the `20260430_193523` training run.

**Implication:** Training and inference are drawn from the same camera system
(WingSCapes WSCT), so pure domain gap is not the primary concern.  The false
negative on `13_WSCT0033.JPG` (CH-08) must be explained by within-distribution
variability (pose, occlusion, lighting, crop quality from SAHI) rather than a
cross-domain shift.  The iNaturalist dataset remains available as a potential
supplement if the camera-trap training set proves insufficient.

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

---

## CH-08 — Fine-tuned classifier still misclassifies the real hummingbird (domain gap confirmed)

**Context:** Run `20260430_193523` fine-tuned EfficientNetV2-S on labeled
WingSCapes camera-trap crops in `data/interim/train_data/` (1176 hummingbird +
2518 other in train; 294 + 629 in val). Hyperparameters:
`lr=0.0012`, `batch_size=128`, `epochs=50` (early-stopped at epoch 12,
`best_epoch=7`), `weight_decay=1e-4`, `patience=5`.

**Validation metrics (W&B, run `lnjepxc0`):**

| Metric | Value at best epoch 7 |
|---|---|
| val/loss | 0.0804 |
| val/accuracy | ~0.990 |
| val/f1 | ~0.980 |
| val/precision | ~0.950 |
| val/recall | ~0.990 |

**Training dynamics:** Two-phase schedule clearly visible in curves — head-only
training (epochs 1–5) produced modest gains; LR reduction at epoch 6 (full
fine-tune, all layers unfrozen) produced a sharp accuracy jump from ~0.94 to
~0.99 in one step. Loss converged at ~0.08 and plateaued; early stopping fired
after epoch 12.

**Real-image inference (`13_WSCT0033.JPG`, 2026-05-01):**
- `detection_confidence` (YOLO+SAHI): 0.364 — marginal but above 0.3 threshold
- `classification_confidence`: **0.362** — below 0.5 threshold → `is_hummingbird: False`
- Ground truth: **1 hummingbird** present
- Outcome: **false negative** despite 99 % validation accuracy

**Root cause analysis — confirmed by crop inspection (2026-05-01):**

The actual inference crop fed to the classifier was saved to
`reports/figures/inference_crop_0.jpg` and compared visually against a
representative training crop (`reports/figures/train_crop_example.jpg`).
The difference is stark:

| | Detection crop (inference) | Training crops |
|---|---|---|
| Source bbox | (6144, 2682, 6352, 3322) | Roboflow YOLO annotations |
| Bbox dimensions | **208 × 640 px** | Typically ~300–600 × 200–400 px |
| Aspect ratio | **0.325** (very tall, narrow) | ~1.0–2.0 (roughly square or wide) |
| After 224×224 resize | ~3× horizontal stretch applied | Minimal distortion |
| Visual appearance | Heavily smeared, unrecognizable | Clear bird at feeder |

The SAHI detection returned a 208 × 640 px bounding box — a narrow vertical
sliver, likely clipping only part of the bird's body (not wings).  `crop_and_resize()`
squashes this non-square region to 224 × 224 with no aspect-ratio correction,
applying a ~3× horizontal expansion.  The result is a heavily distorted image
that bears no resemblance to the clean, well-proportioned crops the model was
trained on.  The classifier correctly rejects it as "not a hummingbird."

For comparison, the ground-truth annotation covers 4130 × 2350 px (AR 1.76),
which would produce a recognisable crop.  The SAHI detection and GT box
overlap spatially but the detected box captures only a thin slice of the scene.

**This is a detector quality problem, not a classifier generalisation problem.**
The train/val accuracy re-check (100 % on all 4 subsets, mean conf 0.998/0.013)
confirms the classifier is well-calibrated when given properly shaped crops.

**Fix options (in order of impact):**
1. **Aspect-ratio-preserving resize** in `crop_and_resize()`: pad the shorter
   dimension to a square before resizing, so distortion is eliminated regardless
   of detection shape.
2. **Improve SAHI detection quality**: tune `nms_iou_threshold` and
   `confidence_threshold` so SAHI merges overlapping tile detections into a
   tighter bounding box around the actual bird body.
3. **Augment training with distorted crops**: add `RandomAffine` with aggressive
   scale/shear to make the classifier robust to imperfect bounding boxes.
   This is lower priority — fixing the crop geometry is cleaner.

**Also noted:** Integration test bugs were discovered and fixed during this
evaluation:
- `process_image()` returns a list of records (one per bird) but tests indexed
  it as a single dict → fixed to `result[0][key]`.
- `is_hummingbird` field was `int` (0/1) instead of `bool` (False/True) in
  `pipeline.py` → fixed with `bool(label)`.
All 10 integration tests pass post-fix.

**Resolution (2026-05-01):**

1. ✅ `crop_and_resize()` now letterboxes non-square bboxes to a square (padded
   with per-channel mean colour) before the `cv2.resize` call.  The 208×640
   crop is centred in a 640×640 canvas and then resized — ~3× stretch eliminated.
2. ✅ Re-ran pipeline on `13_WSCT0033.JPG` post-fix:
   - `classification_confidence`: **0.994** (was 0.362) → `hummingbird: True` ✓
   - All 10 integration tests pass.
3. ✅ Tile size increase: investigated and **not recommended** as the primary fix
   (see CH-09 — the test bird fills any practical tile regardless of size).
4. Collect additional annotated camera-trap frames for systematic evaluation.

---

## CH-09 — SAHI NMS cannot merge large-object tile detections; tile size increase unhelpful

**Context:** SAHI merges per-tile detections via IoU-based NMS
(`nms_iou_threshold=0.5`).  This works well when a small object appears in the
overlap region of two adjacent tiles (high IoU between their detections).  It
fails for objects **larger than the tile size**.

**Measured geometry (`13_WSCT0033.JPG`, bird at 4130×2350 px):**

The GT bird occupies 47 % of image width and 50 % of image height (bird
exceptionally close to camera).  With `slice_size=1280, overlap_ratio=0.2`
(stride 1024 px):

```
IoU between adjacent full-tile detections ≈ 0.111
NMS merge threshold: 0.5
Result: detections are never merged — each tile fires independently
```

The bird covers 20 tiles (8 fully, 12 partially).  YOLO sees the bird filling
the entire tile in every case — no background context, no full silhouette —
and assigns low or no confidence.  Only one marginal edge-of-tile detection
(208×640 px sliver, conf=0.364) survived the 0.3 threshold.

**Tile size investigation:**

| slice_size | tiles | best tile bird coverage | action |
|---|---|---|---|
| 1280 | 45 | 17 % | current config |
| 2048 | 18 | 30 % | marginal gain |
| 3072 | 8 | 68 % | best coverage, but bird still fills tile |
| 4096 | 6 | 53 % | less efficient, stride artefacts |

At every tile size the bird is always larger than or equal to the tile, so YOLO
always sees the bird filling the entire frame.  Increasing tile size does not
solve the detection problem; it only reduces tile count and inference time.

**Root cause distinction from CH-01:** CH-01 (small object loss) applies to
normal feeder-distance birds (~200–500 px).  CH-09 applies to the opposite
regime: a bird so close to the camera that it is larger than the YOLO+SAHI
detection paradigm is designed for.  SAHI was designed for small objects
in large images; this bird is a large object in a large image.

**Consequences and mitigations:**
- The letterbox fix in `crop_and_resize()` (CH-08) means the marginal 0.364
  confidence sliver detection now produces a usable crop for the classifier.
  This is sufficient for the current test image.
- For systematic robustness to very-large-bird frames, options are:
  1. ✅ **Direct-pass first**: `BirdDetector.detect()` now always runs a direct
     full-image YOLO pass before SAHI tiling.  If any birds are found in the
     direct pass, SAHI is skipped.  This handles large near-camera birds without
     tiling overhead, and falls back to SAHI for small distant birds.
     Implemented 2026-05-01.
  2. **Fine-tune YOLO** on WingSCapes crops so it learns to detect birds at all
     scales including full-frame fills.
  3. **Accept as edge case**: if normal feeder images have birds at typical
     distance (~200–500 px), this close-camera case may be rare in practice.
