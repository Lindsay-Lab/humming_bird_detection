"""Tests for humming_bird_detection.data.metadata."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from humming_bird_detection.data.metadata import (
    PhotoMetadata,
    extract_margin,
    parse_metadata,
)


@pytest.fixture
def margin_image() -> np.ndarray:
    """500×272 BGR image with rendered camera metadata text."""
    img = Image.new("RGB", (500, 272), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((10, 100), "72 F  CAMERA 3  07 APR 2025  06:32 AM", fill=(255, 255, 255))
    # PIL is RGB, OpenCV expects BGR
    arr = np.array(img)[:, :, ::-1].copy()
    return arr


# ---------------------------------------------------------------------------
# extract_margin
# ---------------------------------------------------------------------------


def test_extract_margin_shape():
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    margin = extract_margin(image, margin_px=272)
    assert margin.shape == (272, 800, 3)


# ---------------------------------------------------------------------------
# parse_metadata — happy path (mocked EasyOCR)
# ---------------------------------------------------------------------------


def test_parse_metadata_fields(margin_image):
    ocr_text = "72 F  CAMERA 3  07 APR 2025  06:32 AM"
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ocr_text.split("  ")

    with patch("humming_bird_detection.data.metadata._reader", mock_reader):
        result = parse_metadata("test.jpg", margin_image)

    assert isinstance(result, PhotoMetadata)
    assert result.temperature == 72.0
    assert result.camera_id is not None and "CAMERA" in result.camera_id
    assert result.date is not None
    assert result.time is not None


# ---------------------------------------------------------------------------
# parse_metadata — all fields missing
# ---------------------------------------------------------------------------


def test_parse_metadata_missing_field():
    blank = np.zeros((272, 500, 3), dtype=np.uint8)
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []

    with patch("humming_bird_detection.data.metadata._reader", mock_reader):
        result = parse_metadata("blank.jpg", blank)

    assert result.temperature is None
    assert result.camera_id is None
    assert result.date is None
    assert result.time is None
