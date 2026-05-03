"""Tests for humming_bird_detection.models.detector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from humming_bird_detection.models.detector import BirdDetector, Detection, build_detector


def _make_boxes(rows: list[list[float]]) -> MagicMock:
    """Build a mock ultralytics Boxes-like object from a list of [x1,y1,x2,y2,conf,cls] rows."""
    mock_boxes = MagicMock()
    if rows:
        mock_boxes.data = torch.tensor(rows, dtype=torch.float32)
    else:
        mock_boxes.data = torch.zeros((0, 6), dtype=torch.float32)
    mock_boxes.__len__ = lambda self: len(rows)
    return mock_boxes


def _make_results(rows: list[list[float]]) -> list[MagicMock]:
    result = MagicMock()
    result.boxes = _make_boxes(rows)
    return [result]


# ---------------------------------------------------------------------------


def test_detect_returns_detections(detector):
    # bird class 14 conf 0.8, non-bird class 2 conf 0.9
    rows = [
        [10.0, 20.0, 100.0, 200.0, 0.8, 14.0],
        [50.0, 60.0, 150.0, 250.0, 0.9, 2.0],
    ]
    detector._model.return_value = _make_results(rows)
    detections = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].confidence == pytest.approx(0.8, abs=1e-4)


def test_detect_empty_on_no_birds(detector):
    detector._model.return_value = _make_results([])
    result = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert result == []


def test_detect_filters_low_confidence(detector):
    # bird class 14 but conf 0.1 < threshold 0.3
    rows = [[10.0, 20.0, 100.0, 200.0, 0.1, 14.0]]
    detector._model.return_value = _make_results(rows)
    result = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert result == []


def test_oom_error_is_reraised(detector):
    """RuntimeError containing 'out of memory' must propagate, not be swallowed."""
    detector._model.side_effect = RuntimeError(
        "CUDA out of memory. Tried to allocate 2.00 GiB."
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))


def test_other_inference_error_returns_empty(detector):
    """Non-OOM RuntimeErrors are caught, logged, and return empty list."""
    detector._model.side_effect = RuntimeError("invalid model format")
    result = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert result == []


def test_unexpected_exception_returns_empty(detector):
    """Non-RuntimeError exceptions are caught and return empty list."""
    detector._model.side_effect = ValueError("unexpected")
    result = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert result == []


def test_sahi_oom_is_reraised(tiled_detector):
    """OOM during SAHI tiling must propagate, not be swallowed."""
    # Direct pass returns nothing so we fall through to SAHI
    empty = MagicMock()
    empty.boxes = None
    tiled_detector._model.return_value = [empty]

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction",
        side_effect=RuntimeError("CUDA out of memory."),
    ):
        with patch.object(tiled_detector, "_get_sahi_model", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="out of memory"):
                tiled_detector.detect(np.zeros((4000, 6000, 3), dtype=np.uint8))


def test_sahi_non_oom_error_returns_empty(tiled_detector):
    """Non-OOM errors during SAHI tiling are caught and return empty list."""
    empty = MagicMock()
    empty.boxes = None
    tiled_detector._model.return_value = [empty]

    with patch(
        "humming_bird_detection.models.detector.get_sliced_prediction",
        side_effect=RuntimeError("model format error"),
    ):
        with patch.object(tiled_detector, "_get_sahi_model", return_value=MagicMock()):
            result = tiled_detector.detect(np.zeros((4000, 6000, 3), dtype=np.uint8))
    assert result == []


def test_invalid_bird_class_id_raises():
    """bird_class_id absent from model's class list must raise ValueError at init."""
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {0: "cat", 1: "dog"}  # no class 14
        mock_yolo_cls.return_value = mock_instance
        with pytest.raises(ValueError, match="bird_class_id"):
            BirdDetector(
                model_path=Path("/fake/model.pt"),
                bird_class_id=14,
            )


def test_wrong_class_name_logs_warning(caplog):
    """bird_class_id present but mapped to a non-'bird' class should log a warning."""
    import logging

    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {14: "fish"}  # id exists but name is wrong
        mock_yolo_cls.return_value = mock_instance
        with caplog.at_level(logging.WARNING):
            BirdDetector(
                model_path=Path("/fake/model.pt"),
                bird_class_id=14,
            )
    assert any("bird_class_id" in r.message for r in caplog.records)


def test_build_detector_factory():
    """build_detector() constructs a BirdDetector from a config dict."""
    cfg = {
        "detector": {
            "model": "yolo11n.pt",
            "bird_class_id": 14,
            "confidence_threshold": 0.3,
        }
    }
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value.names = {14: "bird"}
        d = build_detector(cfg)
    assert isinstance(d, BirdDetector)
    assert d.bird_class_id == 14
    assert d.confidence_threshold == 0.3
    assert d.slice_size is None
