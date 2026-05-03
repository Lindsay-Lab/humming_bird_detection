# Shared pytest fixtures for the hummingbird detection test suite.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from humming_bird_detection.models.detector import BirdDetector


@pytest.fixture
def detector():
    """BirdDetector with no tiling, YOLO mocked."""
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {14: "bird"}
        mock_yolo_cls.return_value = mock_model
        d = BirdDetector(
            model_path=Path("/fake/model.pt"),
            bird_class_id=14,
            confidence_threshold=0.3,
        )
    return d


@pytest.fixture
def tiled_detector():
    """BirdDetector with slice_size=640, YOLO and SAHI both mocked."""
    with patch("humming_bird_detection.models.detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {14: "bird"}
        mock_yolo_cls.return_value = mock_model
        d = BirdDetector(
            model_path=Path("/fake/model.pt"),
            bird_class_id=14,
            confidence_threshold=0.3,
            slice_size=640,
            overlap_ratio=0.2,
        )
    return d
