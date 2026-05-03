"""Tests for humming_bird_detection.workflow.postprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from humming_bird_detection.workflow.postprocessing import (
    build_postprocessors,
    flag_duplicates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = "07 APR 2025"


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(filename: str, time: str, date: str = _DATE, **kwargs) -> dict:
    return {
        "filename": filename,
        "date": date,
        "time": time,
        "detection_confidence": 0.8,
        "is_hummingbird": True,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# flag_duplicates — core logic
# ---------------------------------------------------------------------------


def test_single_row_not_duplicate():
    """A single detection is never a duplicate."""
    df = _make_df([_row("a.jpg", "06:30 AM")])
    result = flag_duplicates(df, window_seconds=15)
    assert (
        result.loc[0, "is_duplicate"] is False
        or result.loc[0, "is_duplicate"] == False
    )


def test_two_rows_same_file_not_duplicate():
    """Multiple birds in the same image share is_duplicate=False."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("a.jpg", "06:30 AM"),
        ]
    )
    result = flag_duplicates(df, window_seconds=15)
    assert list(result["is_duplicate"]) == [False, False]


def test_same_minute_timestamp_second_is_duplicate():
    """Two images with the same minute-precision timestamp: the later (by filename) is a duplicate."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("b.jpg", "06:30 AM"),
        ]
    )
    result = flag_duplicates(df, window_seconds=15)
    # a.jpg sorts first → not dup; b.jpg 0s later → dup
    dup_by_file = result.drop_duplicates("filename").set_index("filename")[
        "is_duplicate"
    ]
    assert dup_by_file["a.jpg"] == False
    assert dup_by_file["b.jpg"] == True


def test_different_minute_outside_window_not_duplicate():
    """Images one minute apart (60s) are NOT duplicates when window=15s."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("b.jpg", "06:31 AM"),
        ]
    )
    result = flag_duplicates(df, window_seconds=15)
    dup_by_file = result.drop_duplicates("filename").set_index("filename")[
        "is_duplicate"
    ]
    assert dup_by_file["a.jpg"] == False
    assert dup_by_file["b.jpg"] == False


def test_different_minute_inside_window_is_duplicate():
    """Images one minute apart ARE duplicates when window ≥ 60s."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("b.jpg", "06:31 AM"),
        ]
    )
    result = flag_duplicates(df, window_seconds=90)
    dup_by_file = result.drop_duplicates("filename").set_index("filename")[
        "is_duplicate"
    ]
    assert dup_by_file["a.jpg"] == False
    assert dup_by_file["b.jpg"] == True


def test_chain_reference_anchored_to_first():
    """Duplicate chain measures all gaps from the first sighting, not the previous duplicate."""
    # a(06:30) → reference; b(06:30) → dup; c(06:31) → 60s from a → NOT dup with window=15
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("b.jpg", "06:30 AM"),
            _row("c.jpg", "06:31 AM"),
        ]
    )
    result = flag_duplicates(df, window_seconds=15)
    dup_by_file = result.drop_duplicates("filename").set_index("filename")[
        "is_duplicate"
    ]
    assert dup_by_file["a.jpg"] == False
    assert dup_by_file["b.jpg"] == True
    assert dup_by_file["c.jpg"] == False  # 60s from anchor a.jpg > 15s window


def test_multiple_birds_same_image_share_flag():
    """All detections from the same image inherit the same is_duplicate flag."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("b.jpg", "06:30 AM"),
            _row("b.jpg", "06:30 AM"),  # second bird in b.jpg
        ]
    )
    result = flag_duplicates(df, window_seconds=15)
    b_rows = result[result["filename"] == "b.jpg"]["is_duplicate"]
    assert b_rows.iloc[0] == True
    assert b_rows.iloc[1] == True


def test_none_timestamp_treated_as_not_duplicate():
    """Rows with missing date/time are conservatively not flagged as duplicates."""
    df = _make_df(
        [
            {"filename": "a.jpg", "date": None, "time": None, "is_hummingbird": True},
            {"filename": "b.jpg", "date": None, "time": None, "is_hummingbird": True},
        ]
    )
    result = flag_duplicates(df, window_seconds=15)
    assert list(result["is_duplicate"]) == [False, False]


def test_empty_dataframe_returns_is_duplicate_column():
    df = pd.DataFrame(columns=["filename", "date", "time"])
    result = flag_duplicates(df, window_seconds=15)
    assert "is_duplicate" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# build_postprocessors
# ---------------------------------------------------------------------------


def test_build_postprocessors_with_deduplication_key():
    cfg = {"deduplication": {"window_seconds": 18}}
    procs = build_postprocessors(cfg)
    assert len(procs) == 1


def test_build_postprocessors_without_deduplication_key():
    cfg = {"image": {"margin_px": 272}}
    procs = build_postprocessors(cfg)
    assert len(procs) == 0


def test_build_postprocessors_produces_working_flag_duplicates():
    cfg = {"deduplication": {"window_seconds": 15}}
    procs = build_postprocessors(cfg)
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM"),
            _row("b.jpg", "06:30 AM"),
        ]
    )
    result = procs[0](df)
    assert "is_duplicate" in result.columns
    dup_by_file = result.drop_duplicates("filename").set_index("filename")[
        "is_duplicate"
    ]
    assert dup_by_file["a.jpg"] == False
    assert dup_by_file["b.jpg"] == True
