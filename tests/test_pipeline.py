"""Tests for humming_bird_detection.workflow.pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import logging

import cv2
import numpy as np
import pytest

from humming_bird_detection.models.detector import Detection
from humming_bird_detection.workflow.pipeline import process_directory, process_image

_CFG = {
    "image": {"margin_px": 10},
    "preprocessing": {"pad_fraction": 0.10},
    "classifier": {"image_size": 224},
}


def _make_mock_image(path: Path, height: int = 50, width: int = 50) -> None:
    """Write a tiny synthetic JPEG to a path."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def mock_detector():
    d = MagicMock()
    d.detect.return_value = []
    return d


@pytest.fixture
def mock_classifier():
    c = MagicMock()
    c.predict.return_value = [(1, 0.92)]
    return c


# ---------------------------------------------------------------------------


def test_process_image_no_birds(tmp_path, mock_detector, mock_classifier):
    img_path = tmp_path / "frame.jpg"
    _make_mock_image(img_path, height=50, width=50)
    mock_detector.detect.return_value = []

    result = process_image(img_path, mock_detector, mock_classifier, _CFG)
    assert result == []


def test_process_image_one_bird(tmp_path, mock_detector, mock_classifier):
    img_path = tmp_path / "frame.jpg"
    _make_mock_image(img_path, height=50, width=50)

    det = Detection(bbox=(0, 0, 20, 20), confidence=0.8)
    mock_detector.detect.return_value = [det]
    mock_classifier.predict.return_value = [(1, 0.92)]

    # patch parse_metadata so we don't load EasyOCR
    with patch("humming_bird_detection.workflow.pipeline.parse_metadata") as mock_meta:
        from humming_bird_detection.data.metadata import PhotoMetadata

        mock_meta.return_value = PhotoMetadata(
            filename="frame.jpg",
            temperature=72.0,
            camera_id="CAMERA 3",
            date="07 APR 2025",
            time="06:32 AM",
        )
        result = process_image(img_path, mock_detector, mock_classifier, _CFG)

    assert len(result) == 1
    record = result[0]
    assert record["bird_index"] == 0
    assert record["detection_confidence"] == pytest.approx(0.8, abs=1e-4)
    assert record["classification_confidence"] == pytest.approx(0.92, abs=1e-4)
    assert record["is_hummingbird"] == 1


def test_process_directory_creates_csv(tmp_path, mock_detector, mock_classifier):
    # Two synthetic images in input dir
    img_dir = tmp_path / "input"
    img_dir.mkdir()
    _make_mock_image(img_dir / "a.jpg")
    _make_mock_image(img_dir / "b.jpg")

    output_csv = tmp_path / "output.csv"

    det = Detection(bbox=(0, 0, 20, 20), confidence=0.8)
    mock_detector.detect.return_value = [det]
    mock_classifier.predict.return_value = [(1, 0.9)]

    with patch("humming_bird_detection.workflow.pipeline.parse_metadata") as mock_meta:
        from humming_bird_detection.data.metadata import PhotoMetadata

        mock_meta.return_value = PhotoMetadata(
            filename="x.jpg",
            temperature=None,
            camera_id=None,
            date=None,
            time=None,
        )
        df = process_directory(
            img_dir, output_csv, mock_detector, mock_classifier, _CFG
        )

    assert output_csv.exists()
    expected_cols = {
        "filename",
        "date",
        "time",
        "camera",
        "bird_index",
        "detection_confidence",
        "classification_confidence",
        "is_hummingbird",
    }
    assert expected_cols.issubset(set(df.columns))


def test_process_image_multiple_birds(tmp_path, mock_detector, mock_classifier):
    """Two birds in one image → two separate records with identical metadata."""
    img_path = tmp_path / "frame.jpg"
    _make_mock_image(img_path, height=50, width=50)

    dets = [
        Detection(bbox=(0, 0, 10, 10), confidence=0.75),
        Detection(bbox=(20, 20, 40, 40), confidence=0.90),
    ]
    mock_detector.detect.return_value = dets
    # One hummingbird (index 0) and one other bird (index 1)
    mock_classifier.predict.return_value = [(1, 0.88), (0, 0.60)]

    with patch("humming_bird_detection.workflow.pipeline.parse_metadata") as mock_meta:
        from humming_bird_detection.data.metadata import PhotoMetadata

        mock_meta.return_value = PhotoMetadata(
            filename="frame.jpg",
            temperature=70.0,
            camera_id="CAMERA 1",
            date="29 APR 2026",
            time="08:00 AM",
        )
        result = process_image(img_path, mock_detector, mock_classifier, _CFG)

    assert len(result) == 2

    # Both rows share the same metadata
    assert result[0]["filename"] == result[1]["filename"]
    assert result[0]["date"] == result[1]["date"]
    assert result[0]["time"] == result[1]["time"]
    assert result[0]["camera"] == result[1]["camera"]

    # Per-bird fields are distinct
    assert result[0]["bird_index"] == 0
    assert result[1]["bird_index"] == 1
    assert result[0]["detection_confidence"] == pytest.approx(0.75, abs=1e-4)
    assert result[1]["detection_confidence"] == pytest.approx(0.90, abs=1e-4)
    assert result[0]["is_hummingbird"] == 1
    assert result[1]["is_hummingbird"] == 0
    assert result[0]["classification_confidence"] == pytest.approx(0.88, abs=1e-4)
    assert result[1]["classification_confidence"] == pytest.approx(0.60, abs=1e-4)


def test_process_directory_multiple_birds_per_image(
    tmp_path, mock_detector, mock_classifier
):
    """Two birds per image × two images → four rows in the CSV."""
    img_dir = tmp_path / "input"
    img_dir.mkdir()
    _make_mock_image(img_dir / "a.jpg")
    _make_mock_image(img_dir / "b.jpg")

    output_csv = tmp_path / "output.csv"

    dets = [
        Detection(bbox=(0, 0, 10, 10), confidence=0.8),
        Detection(bbox=(20, 20, 40, 40), confidence=0.9),
    ]
    mock_detector.detect.return_value = dets
    mock_classifier.predict.return_value = [(1, 0.9), (0, 0.6)]

    with patch("humming_bird_detection.workflow.pipeline.parse_metadata") as mock_meta:
        from humming_bird_detection.data.metadata import PhotoMetadata

        mock_meta.return_value = PhotoMetadata(
            filename="x.jpg",
            temperature=None,
            camera_id=None,
            date=None,
            time=None,
        )
        df = process_directory(
            img_dir, output_csv, mock_detector, mock_classifier, _CFG
        )

    # 2 birds × 2 images = 4 rows
    assert len(df) == 4
    assert list(df["bird_index"]) == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Error-case tests
# ---------------------------------------------------------------------------


def test_process_image_missing_file_returns_empty(
    tmp_path, mock_detector, mock_classifier
):
    """process_image with a path that does not exist should return [] without raising."""
    result = process_image(
        tmp_path / "does_not_exist.jpg", mock_detector, mock_classifier, _CFG
    )
    assert result == []


def test_process_image_missing_file_logs_warning(
    tmp_path, mock_detector, mock_classifier, caplog
):
    """process_image with an unreadable path should emit a WARNING."""
    with caplog.at_level(logging.WARNING):
        process_image(
            tmp_path / "does_not_exist.jpg", mock_detector, mock_classifier, _CFG
        )
    assert any("Could not read" in r.message for r in caplog.records)


def test_process_directory_empty_dir_returns_empty_csv(
    tmp_path, mock_detector, mock_classifier
):
    """process_directory on a directory with no JPEGs should write an empty CSV."""
    img_dir = tmp_path / "empty"
    img_dir.mkdir()
    output_csv = tmp_path / "out.csv"
    df = process_directory(img_dir, output_csv, mock_detector, mock_classifier, _CFG)
    assert output_csv.exists()
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Incremental writes / resume
# ---------------------------------------------------------------------------


def _patched_meta():
    """Context manager patching parse_metadata to avoid loading EasyOCR."""
    from humming_bird_detection.data.metadata import PhotoMetadata

    p = patch("humming_bird_detection.workflow.pipeline.parse_metadata")
    mock_meta = p.start()
    mock_meta.return_value = PhotoMetadata(
        filename="x.jpg", temperature=None, camera_id=None, date=None, time=None
    )
    return p


def test_process_directory_removes_partial_on_success(
    tmp_path, mock_detector, mock_classifier
):
    """A successful run cleans up its .partial.csv sidecar."""
    img_dir = tmp_path / "input"
    img_dir.mkdir()
    _make_mock_image(img_dir / "a.jpg")
    output_csv = tmp_path / "out.csv"

    mock_detector.detect.return_value = [Detection(bbox=(0, 0, 20, 20), confidence=0.8)]
    mock_classifier.predict.return_value = [(1, 0.9)]

    p = _patched_meta()
    try:
        process_directory(img_dir, output_csv, mock_detector, mock_classifier, _CFG)
    finally:
        p.stop()

    partial = tmp_path / "out.partial.csv"
    assert not partial.exists()
    assert output_csv.exists()


def test_process_directory_resume_skips_processed(
    tmp_path, mock_detector, mock_classifier
):
    """With resume=True, images already in the partial sidecar are not reprocessed."""
    img_dir = tmp_path / "input"
    img_dir.mkdir()
    _make_mock_image(img_dir / "a.jpg")
    _make_mock_image(img_dir / "b.jpg")
    output_csv = tmp_path / "out.csv"

    # Pre-seed a partial file as if "a.jpg" was already processed (1 bird).
    from humming_bird_detection.workflow.pipeline import _RECORD_FIELDS

    seed = {
        "filename": "a.jpg",
        "bird_index": 0,
        "detection_confidence": 0.8,
        "classification_confidence": 0.9,
        "is_hummingbird": True,
    }
    partial = tmp_path / "out.partial.csv"
    partial.write_text(
        ",".join(_RECORD_FIELDS)
        + "\n"
        + ",".join(str(seed.get(f, "")) for f in _RECORD_FIELDS)
        + "\n"
    )

    mock_detector.detect.return_value = [Detection(bbox=(0, 0, 20, 20), confidence=0.7)]
    mock_classifier.predict.return_value = [(1, 0.95)]

    p = _patched_meta()
    try:
        df = process_directory(
            img_dir,
            output_csv,
            mock_detector,
            mock_classifier,
            _CFG,
            resume=True,
        )
    finally:
        p.stop()

    # Only b.jpg should have been passed to the detector (a.jpg was resumed).
    assert mock_detector.detect.call_count == 1
    # Both images represented in the final frame.
    assert set(df["filename"]) == {"a.jpg", "b.jpg"}
    assert df.attrs["total_images"] == 2
