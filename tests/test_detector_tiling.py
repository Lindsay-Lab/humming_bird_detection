"""Tests for BirdDetector SAHI tiled inference path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from humming_bird_detection.models.detector import BirdDetector, Detection


def _make_sahi_prediction(category_id: int, conf: float, bbox_xyxy: list[float]):
    """Build a minimal mock SAHI ObjectPrediction."""
    pred = MagicMock()
    pred.category.id = category_id
    pred.score.value = conf
    pred.bbox.to_xyxy.return_value = bbox_xyxy
    return pred


def _make_sliced_result(predictions: list) -> MagicMock:
    result = MagicMock()
    result.object_prediction_list = predictions
    result.sliced_image_list = [MagicMock()] * 4  # pretend 4 tiles
    return result


# ---------------------------------------------------------------------------


def test_tiled_detect_returns_bird_detections(tiled_detector):
    preds = [
        _make_sahi_prediction(14, 0.75, [100.0, 200.0, 400.0, 600.0]),  # bird
        _make_sahi_prediction(41, 0.90, [10.0, 10.0, 300.0, 300.0]),  # cup — filtered
    ]
    sliced_result = _make_sliced_result(preds)

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction",
        return_value=sliced_result,
    ):
        with patch.object(tiled_detector, "_get_sahi_model", return_value=MagicMock()):
            detections = tiled_detector.detect(
                np.zeros((4000, 6000, 3), dtype=np.uint8)
            )

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].confidence == pytest.approx(0.75, abs=1e-4)
    assert detections[0].bbox == (100, 200, 400, 600)


def test_tiled_detect_filters_low_confidence(tiled_detector):
    preds = [
        _make_sahi_prediction(
            14, 0.1, [100.0, 200.0, 400.0, 600.0]
        ),  # bird but conf < 0.3
    ]
    sliced_result = _make_sliced_result(preds)

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction",
        return_value=sliced_result,
    ):
        with patch.object(tiled_detector, "_get_sahi_model", return_value=MagicMock()):
            detections = tiled_detector.detect(
                np.zeros((4000, 6000, 3), dtype=np.uint8)
            )

    assert detections == []


def test_tiled_detect_empty_result(tiled_detector):
    sliced_result = _make_sliced_result([])

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction",
        return_value=sliced_result,
    ):
        with patch.object(tiled_detector, "_get_sahi_model", return_value=MagicMock()):
            detections = tiled_detector.detect(
                np.zeros((4000, 6000, 3), dtype=np.uint8)
            )

    assert detections == []


def test_direct_path_used_when_no_slice_size():
    """When slice_size=None, detect() must use the direct YOLO path, not SAHI."""
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {14: "bird"}
        mock_yolo_cls.return_value = mock_model
        d = BirdDetector(model_path=Path("/fake/model.pt"), slice_size=None)

    mock_model.return_value = [
        MagicMock(
            boxes=MagicMock(
                data=__import__("torch").zeros((0, 6)), __len__=lambda s: 0
            )
        )
    ]

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction"
    ) as mock_sahi:
        d.detect(np.zeros((300, 300, 3), dtype=np.uint8))
        mock_sahi.assert_not_called()


def _make_yolo_result(detections: list[tuple]) -> MagicMock:
    """Build a minimal mock YOLO result whose boxes.data.tolist() returns rows."""
    import torch

    rows = [[x1, y1, x2, y2, conf, cls] for x1, y1, x2, y2, conf, cls in detections]
    boxes = MagicMock()
    boxes.data.tolist.return_value = rows
    boxes.__len__ = lambda s: len(rows)
    result = MagicMock()
    result.boxes = boxes if rows else None
    return result


def test_direct_hit_short_circuits_sahi(tiled_detector):
    """When direct YOLO pass finds a bird, SAHI must NOT be called."""
    yolo_result = _make_yolo_result([(100, 200, 400, 600, 0.85, 14)])
    tiled_detector._model.return_value = [yolo_result]

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction"
    ) as mock_sahi:
        detections = tiled_detector.detect(np.zeros((4000, 6000, 3), dtype=np.uint8))

    mock_sahi.assert_not_called()
    assert len(detections) == 1
    assert detections[0].bbox == (100, 200, 400, 600)
    assert detections[0].confidence == pytest.approx(0.85, abs=1e-4)


def test_direct_miss_falls_back_to_sahi(tiled_detector):
    """When direct YOLO pass finds nothing, SAHI tiling must be called."""
    # Direct pass: no detections (empty boxes)
    empty_result = MagicMock()
    empty_result.boxes = None
    tiled_detector._model.return_value = [empty_result]

    sahi_pred = _make_sahi_prediction(14, 0.72, [50.0, 60.0, 300.0, 400.0])
    sliced_result = _make_sliced_result([sahi_pred])

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction",
        return_value=sliced_result,
    ):
        with patch.object(tiled_detector, "_get_sahi_model", return_value=MagicMock()):
            detections = tiled_detector.detect(
                np.zeros((4000, 6000, 3), dtype=np.uint8)
            )

    assert len(detections) == 1
    assert detections[0].bbox == (50, 60, 300, 400)
