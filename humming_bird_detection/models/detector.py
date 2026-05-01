"""YOLO-based bird detection wrapper with optional SAHI tiled inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from humming_bird_detection.project_paths import project_paths as _project_paths

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float


class BirdDetector:
    def __init__(
        self,
        model_path: str | Path,
        bird_class_id: int = 14,
        confidence_threshold: float = 0.3,
        slice_size: int | None = None,
        overlap_ratio: float = 0.2,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        """Load YOLO model from model_path ready for inference.

        When slice_size is set, inference uses SAHI tiled prediction so that
        small objects in high-resolution images are not lost during YOLO's
        internal downscale.  When slice_size is None the image is passed to
        YOLO directly (faster, suitable for already-cropped / small inputs).

        nms_iou_threshold controls SAHI's post-NMS merge step: detections
        from adjacent tiles whose IoU exceeds this value are collapsed into one
        box.  The default (0.5) matches SAHI's own default.  Lower it (e.g.
        0.3) if two birds in close proximity are being incorrectly merged.
        """
        self.bird_class_id = bird_class_id
        self.confidence_threshold = confidence_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.nms_iou_threshold = nms_iou_threshold
        self._model_path = str(model_path)

        # Direct YOLO model — always initialised so unit tests can mock it.
        self._model = YOLO(self._model_path)

        # Validate that bird_class_id is present in the model's known classes.
        model_names: dict = self._model.names  # {int: str}
        if self.bird_class_id not in model_names:
            raise ValueError(
                f"bird_class_id={self.bird_class_id} is not a valid class index for "
                f"the loaded model (valid indices: 0–{max(model_names)}). "
                "For COCO-trained YOLO models the bird class is 14. "
                "Set bird_class_id explicitly if using a custom-trained model."
            )
        if model_names[self.bird_class_id].lower() != "bird":
            logger.warning(
                "bird_class_id=%d maps to class '%s' (expected 'bird'). "
                "Verify that bird_class_id matches the model's class indexing.",
                self.bird_class_id,
                model_names[self.bird_class_id],
            )

        # SAHI wrapper — initialised lazily on first tiled call so that tests
        # using slice_size=None never touch the SAHI stack.
        self._sahi_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return bird detections.

        When slice_size is set, first tries a direct full-image YOLO pass.  If
        any birds are found, those detections are returned immediately (no
        tiling overhead).  Only if the direct pass finds nothing does the
        detector fall back to SAHI tiled inference.

        This two-stage approach handles both regimes:
        - Large / near-camera birds: YOLO sees them at full resolution even
          after its internal 640 px downscale (a 4130×2350 px bird maps to
          ~303×172 px in YOLO's input space — well above its detection floor).
        - Small / far-camera birds: YOLO misses them at full resolution
          (~15 px after downscale); SAHI tiling recovers them.

        When slice_size is None, only direct inference is used.
        """
        if self.slice_size is None:
            return self._detect_direct(image)

        direct = self._detect_direct(image)
        if direct:
            logger.info(
                "Direct YOLO pass: %d bird(s) found — skipping SAHI tiling",
                len(direct),
            )
            return direct

        logger.info("Direct YOLO pass: no birds found — running SAHI tiled inference")
        return self._detect_tiled(image)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_direct(self, image: np.ndarray) -> list[Detection]:
        """Single-pass YOLO inference on the full (or pre-cropped) image."""
        try:
            results = self._model(image, verbose=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise
            logger.exception("YOLO inference failed")
            return []
        except Exception:
            logger.exception("YOLO inference failed (unexpected error)")
            return []

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for row in boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = row[:6]
            if int(cls) == self.bird_class_id and conf >= self.confidence_threshold:
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                    )
                )
        return detections

    def _get_sahi_model(self):
        """Lazily build (and cache) the SAHI AutoDetectionModel."""
        if self._sahi_model is None:
            from sahi import AutoDetectionModel  # noqa: PLC0415

            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self._model_path,
                confidence_threshold=self.confidence_threshold,
                device="cpu",
            )
        return self._sahi_model

    def _detect_tiled(self, image: np.ndarray) -> list[Detection]:
        """SAHI sliced prediction: tile the image, run YOLO per tile, merge."""
        try:
            result = get_sliced_prediction(
                image,
                self._get_sahi_model(),
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                postprocess_match_threshold=self.nms_iou_threshold,
                verbose=0,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise
            logger.exception("SAHI tiled inference failed")
            return []
        except Exception:
            logger.exception("SAHI tiled inference failed (unexpected error)")
            return []

        detections: list[Detection] = []
        for pred in result.object_prediction_list:
            # SAHI category IDs mirror the underlying model's class indices.
            if pred.category.id != self.bird_class_id:
                continue
            if pred.score.value < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(pred.score.value),
                )
            )
        logger.info(
            "SAHI tiled detection: %d tiles → %d bird detections",
            (
                len(result.sliced_image_list)
                if hasattr(result, "sliced_image_list")
                else -1
            ),
            len(detections),
        )
        return detections


def build_detector(cfg: dict) -> BirdDetector:
    """Instantiate BirdDetector from a pipeline config dict."""
    det_cfg = cfg["detector"]
    model_path = _project_paths.models / "YOLO" / det_cfg["model"]
    return BirdDetector(
        model_path=model_path,
        bird_class_id=det_cfg["bird_class_id"],
        confidence_threshold=det_cfg["confidence_threshold"],
        slice_size=det_cfg.get("slice_size"),
        overlap_ratio=det_cfg.get("overlap_ratio", 0.2),
        nms_iou_threshold=det_cfg.get("nms_iou_threshold", 0.5),
    )
