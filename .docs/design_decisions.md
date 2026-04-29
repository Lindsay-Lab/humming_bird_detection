# Design Decisions

This document records the rationale behind key technical choices made during
the implementation of the hummingbird detection pipeline.  Each entry includes
the alternatives considered and the reason for the final choice.

---

## DD-01 — YOLO11n as the detection backbone

**Decision:** Use `yolo11n` (nano variant) as the off-the-shelf bird detector.

**Alternatives considered:**
- YOLO11s / YOLO11m: higher accuracy, 2–5× slower inference.
- Two-stage detectors (Faster R-CNN, DETR): better recall on small objects but
  much larger memory footprint and no off-the-shelf COCO weights as compact.
- Background subtraction / frame differencing: fast but produces pixel masks,
  not bounding boxes; cannot distinguish bird species.

**Rationale:** yolo11n ships with COCO weights that include class 14 ("bird"),
enabling zero-shot detection with no training data.  It fits on a CPU, and its
accuracy gap vs. larger variants is compensated by the SAHI tiling strategy
(see DD-03).  The `model` key in `pipeline.yaml` is `# TUNABLE` so the
researcher can swap to yolo11s/m without code changes.

---

## DD-02 — EfficientNetV2-S as the classification backbone

**Decision:** Replace the final classifier head of
`torchvision.models.efficientnet_v2_s` with
`nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))` and train with
`BCEWithLogitsLoss`.

**Alternatives considered:**
- ResNet-50: well-understood, but EfficientNetV2-S achieves comparable ImageNet
  accuracy at ~30% fewer parameters.
- MobileNetV3: slightly smaller, but accuracy on fine-grained bird crops is
  noticeably weaker.
- Multi-class softmax (hummingbird / other-bird / no-bird): would require a
  three-class head and balanced data for all three classes.  Binary framing
  (is-hummingbird?) simplifies labeling and loss.

**Rationale:** Binary BCEWithLogitsLoss with `pos_weight = n_neg / n_pos`
handles class imbalance (expected: many more non-hummingbird frames) without
oversampling.  The single sigmoid output is easily thresholded and the
confidence value is directly interpretable.

---

## DD-03 — SAHI tiled inference for high-resolution images

**Decision:** Wrap YOLO inference with SAHI `get_sliced_prediction` when
`detector.slice_size` is set in config; fall back to single-pass YOLO when
`slice_size: null`.

**Problem that drove this decision:**
The test image (`13_WSCT0033.JPG`) is 8704×4896 px.  YOLO11n internally
downscales all input to 640×640.  At this scale each output pixel covers ~14
input pixels, making a ~200×200 px hummingbird appear as only ~14×14 px —
below YOLO's reliable detection threshold.

Direct inference results on the test image:
- Detected: class 41 ("cup"), conf=0.55, bbox (1918,13)–(4531,3126)
  → the feeder, **not the bird**.
- Missed: the hand-labeled hummingbird at bbox center (6635, 3435).

SAHI tiling (1280 px tiles, 20% overlap) results:
- Detected: 1 bird (class 14) in the correct region ✓

**Alternatives considered:**
- Increase YOLO `imgsz` to 3840: still 2–3× too low; GPU memory cost rises
  quadratically; ~20–30× slower on CPU.
- Fine-tune YOLO on Trochilidae data: best long-term accuracy, but requires
  labeled training data and a GPU training run — deferred.
- Accept "cup" / "vase" detections as feeder-proxy, pass to classifier: works
  only when bird is at the same feeder YOLO sees; fragile.

**Trade-off:** SAHI at `slice_size=1280` generates ≈56 tiles per image and
takes ~90 s on a CPU core.  This is acceptable for batch overnight processing
but prohibitive for real-time use.  `slice_size` is configurable; setting it
to `null` restores sub-3-second direct inference for smaller images or
pre-cropped inputs.

---

## DD-04 — Dual inference paths inside `BirdDetector`

**Decision:** A single `BirdDetector` class handles both direct and tiled
inference, dispatching on `self.slice_size`.

**Rationale:** Keeps the public API (`detector.detect(image)`) identical
regardless of inference strategy.  Pipeline code, tests, and the future CLI
never need to check which mode is active.  Tests for the direct path
(test_detector.py) are completely unaffected by the SAHI dependency.

---

## DD-05 — Lazy SAHI model initialisation

**Decision:** `_sahi_model` is `None` at construction time and built on the
first tiled call via `_get_sahi_model()`.

**Rationale:** `AutoDetectionModel.from_pretrained` is slow (loads model twice
— once into the native Ultralytics wrapper, once into SAHI's wrapper).  Most
unit tests use `slice_size=None` and never trigger this path.  Eagerly
constructing it would slow down every test collection and every non-tiled
detector instantiation.

---

## DD-06 — Temperature regex tolerates OCR misreads

**Decision:** Temperature pattern changed from `(\d+)\s*F\b` to
`(\d+)\s*[F'\`"]` to match apostrophe/quote OCR artefacts.

**Observation from real image:** The WingSCapes camera renders `86 F` in the
margin, but EasyOCR (CPU, no GPU) reads `86 '` — the capital `F` is
near-systematically misidentified as an apostrophe or single-quote in the
compressed low-light rendering of the metadata strip.

**Risk:** The broader pattern would also match numbers like `5'` (feet) if
such strings appear in the OCR output.  Acceptable because the full OCR text
is short (≈10 tokens from the metadata strip) and no other field contains
standalone digits followed by a quote.

---

## DD-07 — Time regex tolerates space-padded colons

**Decision:** Time pattern changed from `(\d{1,2}:\d{2}\s*[AP]M)` to
`(\d{1,2}\s*:\s*\d{2}\s*[AP]M)`.

**Observation:** EasyOCR returns `04 : 18PM` with spaces around the colon for
the WingSCapes timestamp.  The original tight pattern failed to match.

---

## DD-08 — Two-phase fine-tuning strategy for `train_model()`

**Decision:** Training proceeds in two phases:
- Phase 1 (epochs 1–5): the EfficientNetV2-S feature extractor (`self._model.features`)
  is frozen; only the classification head `(Dropout + Linear)` is updated at the
  full learning rate.
- Phase 2 (remaining epochs): all parameters unfrozen; learning rate reduced to
  `lr / 10`; `CosineAnnealingLR` applied over the remaining epoch budget.

**Alternatives considered:**
- Train all layers from epoch 1 at full LR: with only ~215 training images (80 %
  of 269 iNaturalist crops), the backbone has far more capacity than the data can
  support; early full fine-tuning would corrupt the well-calibrated ImageNet
  features before the head has converged.
- LoRA / adapter layers: unnecessary overhead for a two-class binary problem on a
  small-to-medium backbone.

**Rationale:** Head-only phase converges the final linear layer at low cost, then
full fine-tuning with a reduced LR adjusts the backbone's higher-level features
toward the hummingbird crop distribution.  CosineAnnealingLR prevents sharp
decay steps and has been shown to outperform step-decay schedules in small-data
regimes.

---

## DD-09 — iNaturalist API as source for "other" class training images

**Decision:** Non-hummingbird ("other") training images are downloaded from the
iNaturalist public API using the existing `load_inaturalist.py` script with
alternate `--taxon_name` arguments (e.g. Tyrannidae, Corvidae, Parulidae).

**Alternatives considered:**
- CUB-200-2011 / NABirds: both require downloading a 1–11 GB archive and
  reverse-engineering species-to-class mappings; domain characteristics are
  similar to iNaturalist (clean subject photos), so they offer no advantage.
- YOLO+SAHI negatives from camera-trap images: best domain match (exactly what
  the classifier sees at inference time), but only one camera-trap image currently
  exists in the repo (and it contains a hummingbird).

**Rationale:** The iNaturalist script is already integrated, requires no
additional dependencies, and downloads CC-BY/CC0 licensed images at the
"original" resolution.  Sourcing ~300 images from 3–4 non-hummingbird families
yields adequate class balance.  Domain-matched camera-trap negatives should be
added when more camera footage becomes available.

**Label mapping note:** `torchvision.datasets.ImageFolder` sorts subdirectories
alphabetically; the layout `{split}/hummingbird/` and `{split}/other/` yields
`class_to_idx = {'hummingbird': 0, 'other': 1}`.  `train_model()` explicitly
remaps these so that hummingbird samples receive `target = 1` in
`BCEWithLogitsLoss`, matching the inference convention (`label = 1` → hummingbird).

---

## DD-10 — Exposing SAHI's NMS IoU threshold for multi-bird scenes

**Decision:** Add `nms_iou_threshold` (default 0.5) as an explicit parameter
to `BirdDetector.__init__`, to the `build_detector` factory, and to the
`detector` section of `pipeline.yaml`.  It is passed to
`get_sliced_prediction` as `postprocess_match_threshold`.

**Background:** SAHI's tiled inference produces one YOLO prediction per tile.
Adjacent tiles overlap (controlled by `overlap_ratio`), so the same bird will
be detected in multiple tiles.  SAHI's post-processing step runs NMS across
all tile predictions: any two boxes whose Intersection over Union (IoU) exceeds
`postprocess_match_threshold` are merged into a single box (the one with the
higher score is kept).

**Risk for multi-bird scenes:** If two birds are very close — for example two
hummingbirds hovering at adjacent feeder ports — their bounding boxes may
overlap substantially.  With the default threshold (0.5) such boxes could be
collapsed into one, causing the pipeline to record a single detection where
there are actually two birds.

**Why 0.5 is a safe default:** Most feeder photos show at most one bird at a
time; the overlap between a tile-boundary duplicate and the canonical box is
typically > 0.7.  Setting the threshold lower than 0.5 risks retaining the
tile-boundary duplicate as a spurious second detection for a single bird.

**Guidance for adjustment:** If field review shows that two birds close
together are regularly merged into one detection, lower `nms_iou_threshold` to
0.3–0.4.  If many duplicate detections appear for single birds (especially near
tile boundaries), raise it to 0.6–0.7.  The correct value depends on how
densely birds cluster around the monitored feeder.
