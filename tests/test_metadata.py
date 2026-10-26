"""Tests for humming_bird_detection.data.metadata."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from humming_bird_detection.data.metadata import (
    PhotoMetadata,
    _normalize_time,
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


# ---------------------------------------------------------------------------
# parse_metadata — real-world OCR variants
# ---------------------------------------------------------------------------
#
# EasyOCR misreads the Wingscapes "°F" glyph as %, P, o, etc., and emits
# inconsistent spacing around the time colon and meridiem.  These strings are
# verbatim captures from data/raw/test_data; the parser must recover the
# temperature, camera, date, and a strptime-compatible time from each.


@pytest.mark.parametrize(
    "ocr_text, exp_temp, exp_cam, exp_date, exp_time",
    [
        (
            'WINGSCAPES" 107 % CAMERA 3 23 JAN 2024 02 : 28PM',
            107.0,
            "CAMERA 3",
            "23 JAN 2024",
            "02:28 PM",
        ),
        (
            'WINGSCAPES" 89 % CAMERA 3 25 JAN 2024 10 :55AM',
            89.0,
            "CAMERA 3",
            "25 JAN 2024",
            "10:55 AM",
        ),
        (
            'WINGSCAPES" 100 P CAMERA 3 26 JAN 2024 12 :21PM',
            100.0,
            "CAMERA 3",
            "26 JAN 2024",
            "12:21 PM",
        ),
        (
            # Leading battery/degree glyph "0" must not be taken as the temp.
            'WINGSCAPES" 0 82 % CAMERA 4 17 JAN 2024 06 : 02PM',
            82.0,
            "CAMERA 4",
            "17 JAN 2024",
            "06:02 PM",
        ),
        (
            "72 F  CAMERA 3  07 APR 2025  06:32 AM",
            72.0,
            "CAMERA 3",
            "07 APR 2025",
            "06:32 AM",
        ),
    ],
)
def test_parse_metadata_real_world_ocr(
    ocr_text, exp_temp, exp_cam, exp_date, exp_time
):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [ocr_text]

    with patch("humming_bird_detection.data.metadata._reader", mock_reader):
        result = parse_metadata("real.jpg", np.zeros((272, 500, 3), dtype=np.uint8))

    assert result.temperature == exp_temp
    assert result.camera_id == exp_cam
    assert result.date == exp_date
    assert result.time == exp_time


def test_parse_metadata_time_is_strptime_compatible():
    """The normalised time must parse with the downstream date/time format."""
    from datetime import datetime

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        'WINGSCAPES" 107 % CAMERA 3 23 JAN 2024 02 : 28PM'
    ]
    with patch("humming_bird_detection.data.metadata._reader", mock_reader):
        result = parse_metadata("real.jpg", np.zeros((272, 500, 3), dtype=np.uint8))

    parsed = datetime.strptime(
        f"{result.date} {result.time}", "%d %b %Y %I:%M %p"
    )
    assert parsed == datetime(2024, 1, 23, 14, 28)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("02 : 28PM", "02:28 PM"),
        ("10 :55AM", "10:55 AM"),
        ("6:32 AM", "06:32 AM"),
        ("06:32 AM", "06:32 AM"),
        (None, None),
        ("garbage", None),
    ],
)
def test_normalize_time(raw, expected):
    assert _normalize_time(raw) == expected
