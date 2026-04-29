"""Tests for humming_bird_detection.models.detector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from humming_bird_detection.models.detector import BirdDetector, Detection


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


@pytest.fixture
def detector():
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock()
        d = BirdDetector(
            model_path=Path("/fake/model.pt"),
            bird_class_id=14,
            confidence_threshold=0.3,
        )
    return d


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
