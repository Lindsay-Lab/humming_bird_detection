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


@pytest.fixture
def tiled_detector():
    """BirdDetector with slice_size=640, YOLO and SAHI both mocked."""
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock()
        d = BirdDetector(
            model_path=Path("/fake/model.pt"),
            bird_class_id=14,
            confidence_threshold=0.3,
            slice_size=640,
            overlap_ratio=0.2,
        )
    return d


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
