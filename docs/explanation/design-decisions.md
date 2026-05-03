# Design decisions

## Why EfficientNetV2-S?

EfficientNetV2-S offers a favourable accuracy/latency trade-off for binary classification on moderately sized training sets. Compared to ResNet-50 it is faster on CPU (important for field deployments) and has fewer parameters to overfit on datasets with hundreds rather than thousands of training images.

Pre-trained ImageNet weights give a strong feature initialisation, so only a small fine-tuning dataset is needed to achieve good separation between hummingbirds and other birds.

## Why a binary head with BCE loss?

The classification task is fundamentally binary: hummingbird vs. everything else. Using a single sigmoid output with `BCEWithLogitsLoss` directly optimises the two-class decision boundary. A two-neuron softmax head with cross-entropy would add a redundant parameter and impose an irrelevant constraint (probabilities sum to 1 for classes that are not mutually exclusive in the real world).

`pos_weight` in the BCE loss compensates automatically for class imbalance: it is computed at training time as `n_other / n_hummingbird` and passed to the loss function.

## Why YOLO11n?

- **Offline operation**: camera-trap field work often occurs without internet access. YOLO11n runs entirely locally.
- **Speed**: inference on a CPU takes < 0.5 s per image, allowing batch processing of large data collections overnight.
- **COCO class 14 coverage**: the "bird" class in COCO covers virtually all species, so no detector fine-tuning is needed.

YOLO11n is the nano variant — not the most accurate, but accurate enough for conspicuous bird subjects and fast enough for CPU-only deployment.

## Why INTER_AREA for downscaling?

OpenCV's `INTER_AREA` averages the pixel block when shrinking an image, which reduces aliasing compared to `INTER_LINEAR` or `INTER_CUBIC`. Camera-trap images are often 4000 × 3000 px and are downscaled significantly before inference; INTER_AREA preserves fine detail (feathers, colour patterns) better in this regime.

## Why ImageNet normalisation?

The EfficientNetV2-S backbone was pre-trained on ImageNet with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Normalising inputs with the same statistics ensures that the pre-trained weights operate in their expected numeric range. Deviating from these values would degrade the feature representations inherited from ImageNet pre-training and require more epochs of fine-tuning to recover.

## Why `pipeline.yaml` rather than code-level defaults?

Separating configuration from code allows:

1. **Reproducibility**: committing a YAML file to version control captures the exact settings used for a given analysis run.
2. **Deployment flexibility**: changing `margin_px` or `confidence_threshold` for a new camera model does not require editing Python.
3. **Validation**: `_validate_config()` checks for required keys at load time and fails loudly rather than silently using wrong defaults.

## Why letterbox padding before classifier resize?

When a bounding box is taller than it is wide (or vice versa), naively resizing it to a square (224 × 224) applies a non-uniform scale, stretching the content. During evaluation, a SAHI detection returned a 208 × 640 px bounding box (a narrow vertical sliver). Direct resize to 224 × 224 applied a ~3 × horizontal expansion, producing a smeared, unrecognisable image that the classifier correctly rejected as "not a hummingbird" — causing a false negative despite 99 % validation accuracy.

The fix is to pad the shorter dimension to match the longer one (using the per-channel mean colour of the crop) before resizing. This centres the content in a square canvas, so the resize is uniform and aspect ratio is preserved. Per-channel mean fill avoids a hard black or white border that would be absent from training crops.

## Why direct YOLO pass before SAHI tiling?

SAHI tiling was designed for *small objects* in large images: it works by dividing the image into tiles small enough that the object is large relative to the tile, then merging per-tile detections with NMS. The NMS merge step requires an IoU of at least `nms_iou_threshold` (default 0.5) between adjacent detections.

For *large objects* (bird much bigger than the tile), adjacent tile detections have very low IoU (≈ 0.11 for a bird spanning four tiles) and are never merged. The bird may only produce marginal detections at the edges of tiles where partial context is visible.

Running a direct full-image YOLO pass first solves this regime: the large bird maps to ≈ 300 × 170 px at YOLO's 640 px input — well above the detection floor — so it is detected cleanly in a single pass. The SAHI fallback then handles the opposite regime (small distant birds, ≈ 15 px after YOLO downscale) when the direct pass finds nothing.

This two-pass approach is strictly more capable than either strategy alone, at the cost of one extra YOLO inference call (< 0.5 s on CPU) before the ~34 s tiling step.
