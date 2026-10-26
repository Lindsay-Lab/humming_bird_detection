"""Tests for humming_bird_detection.workflow.postprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from humming_bird_detection.workflow.postprocessing import (
    aggregate_to_photos,
    build_postprocessors,
    flag_duplicates,
    flag_suspected_duplicates,
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


def test_build_postprocessors_with_suspected_duplicates_key():
    cfg = {"suspected_duplicates": {"window_seconds": 60}}
    procs = build_postprocessors(cfg)
    assert len(procs) == 1


def test_build_postprocessors_with_both_keys():
    cfg = {
        "deduplication": {"window_seconds": 18},
        "suspected_duplicates": {"window_seconds": 60},
    }
    procs = build_postprocessors(cfg)
    assert len(procs) == 2


# ---------------------------------------------------------------------------
# flag_suspected_duplicates
# ---------------------------------------------------------------------------


def test_suspected_dup_single_photo_not_flagged():
    """A single photo in a minute window is never suspected."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    assert result.loc[0, "is_suspected_duplicate"] == False


def test_suspected_dup_two_identical_photos_one_flagged():
    """2 photos same minute, same composition → keep 1, flag 1 (ceil(2/3)=1)."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1"),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    fname_flags = result.drop_duplicates("filename").set_index("filename")[
        "is_suspected_duplicate"
    ]
    # a.jpg sorts first → kept; b.jpg → suspected
    assert fname_flags["a.jpg"] == False
    assert fname_flags["b.jpg"] == True


def test_suspected_dup_three_same_minute_keep_one():
    """3 photos same minute, same composition → keep 1, flag 2 (ceil(3/3)=1)."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("c.jpg", "06:30 AM", camera="CAMERA 1"),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    fname_flags = result.drop_duplicates("filename").set_index("filename")[
        "is_suspected_duplicate"
    ]
    assert fname_flags["a.jpg"] == False  # kept
    assert fname_flags["b.jpg"] == True
    assert fname_flags["c.jpg"] == True


def test_suspected_dup_four_keep_two():
    """4 photos same minute/composition → keep 2, flag 2 (ceil(4/3)=2)."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("c.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("d.jpg", "06:30 AM", camera="CAMERA 1"),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    n_suspected = result.drop_duplicates("filename")["is_suspected_duplicate"].sum()
    assert n_suspected == 2


def test_suspected_dup_seven_keep_three():
    """7 photos same minute/composition → keep 3, flag 4 (ceil(7/3)=3)."""
    fnames = [f"{chr(97 + i)}.jpg" for i in range(7)]
    df = _make_df(
        [_row(f, "06:30 AM", camera="CAMERA 1") for f in fnames]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    n_suspected = result.drop_duplicates("filename")["is_suspected_duplicate"].sum()
    assert n_suspected == 4


def test_suspected_dup_different_composition_not_grouped():
    """Photos with different bird compositions are NOT grouped together."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1", is_hummingbird=True),
            _row("a1.jpg", "06:30 AM", camera="CAMERA 1", is_hummingbird=False),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1", is_hummingbird=True),
            _row("b1.jpg", "06:30 AM", camera="CAMERA 1", is_hummingbird=False),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    # a.jpg has 1 HB, a1.jpg is in the same photo (different bird)
    # a.jpg composition: HB1_OT1, b.jpg: HB1_OT1 → these ARE grouped
    # But a.jpg and b.jpg are same composition → 2 photos → keep 1
    fname_flags = result.drop_duplicates("filename").set_index("filename")[
        "is_suspected_duplicate"
    ]
    # a.jpg sorts first → kept
    assert fname_flags["a.jpg"] == False
    assert fname_flags["b.jpg"] == True


def test_suspected_dup_different_cameras_independent():
    """Different cameras are deduplicated independently."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("c.jpg", "06:30 AM", camera="CAMERA 2"),
            _row("d.jpg", "06:30 AM", camera="CAMERA 2"),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    fname_flags = result.drop_duplicates("filename").set_index("filename")[
        "is_suspected_duplicate"
    ]
    # CAMERA 1: a.jpg kept, b.jpg suspected
    # CAMERA 2: c.jpg kept, d.jpg suspected
    assert fname_flags["a.jpg"] == False
    assert fname_flags["b.jpg"] == True
    assert fname_flags["c.jpg"] == False
    assert fname_flags["d.jpg"] == True


def test_suspected_dup_different_minutes_not_grouped():
    """Photos in different minutes are not grouped."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:31 AM", camera="CAMERA 1"),
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    assert result["is_suspected_duplicate"].sum() == 0


def test_suspected_dup_all_birds_same_photo_share_flag():
    """All bird detections from the same photo get the same flag."""
    df = _make_df(
        [
            _row("a.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("b.jpg", "06:30 AM", camera="CAMERA 1"),  # second bird in b.jpg
            _row("c.jpg", "06:30 AM", camera="CAMERA 1"),
            _row("c.jpg", "06:30 AM", camera="CAMERA 1"),  # second bird in c.jpg
        ]
    )
    result = flag_suspected_duplicates(df, window_seconds=60)
    b_rows = result[result["filename"] == "b.jpg"]["is_suspected_duplicate"]
    c_rows = result[result["filename"] == "c.jpg"]["is_suspected_duplicate"]
    # a.jpg has 1 bird, b.jpg has 2 birds, c.jpg has 2 birds
    # Compositions: a=HB1_OT0, b=HB2_OT0, c=HB2_OT0
    # b.jpg and c.jpg are same composition in same minute → 2 photos → keep 1
    # b.jpg sorts first → kept, c.jpg → suspected
    assert b_rows.iloc[0] == False  # kept
    assert b_rows.iloc[1] == False
    assert c_rows.iloc[0] == True  # suspected
    assert c_rows.iloc[1] == True


def test_suspected_dup_empty_dataframe():
    df = pd.DataFrame(columns=["filename", "date", "time", "camera", "is_hummingbird"])
    result = flag_suspected_duplicates(df)
    assert "is_suspected_duplicate" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# aggregate_to_photos
# ---------------------------------------------------------------------------


def test_aggregate_to_photos_basic():
    """Basic aggregation: two birds in one photo, one in another."""
    df = pd.DataFrame(
        [
            {
                "filename": "a.jpg",
                "date": "07 APR 2025",
                "time": "06:30 AM",
                "temperature": 72.0,
                "camera": "CAMERA 1",
                "bird_index": 0,
                "detection_confidence": 0.85,
                "classification_confidence": 0.92,
                "is_hummingbird": True,
            },
            {
                "filename": "a.jpg",
                "date": "07 APR 2025",
                "time": "06:30 AM",
                "temperature": 72.0,
                "camera": "CAMERA 1",
                "bird_index": 1,
                "detection_confidence": 0.75,
                "classification_confidence": 0.45,
                "is_hummingbird": False,
            },
            {
                "filename": "b.jpg",
                "date": "07 APR 2025",
                "time": "06:31 AM",
                "temperature": 72.0,
                "camera": "CAMERA 1",
                "bird_index": 0,
                "detection_confidence": 0.90,
                "classification_confidence": 0.88,
                "is_hummingbird": True,
            },
        ]
    )
    result = aggregate_to_photos(df)
    assert len(result) == 2

    a = result[result["filename"] == "a.jpg"].iloc[0]
    assert a["num_total_birds"] == 2
    assert a["num_hummingbirds"] == 1
    assert a["num_other"] == 1
    assert a["has_hummingbird"] == True
    assert a["mean_detection_conf"] == pytest.approx(0.80)
    assert a["max_detection_conf"] == pytest.approx(0.85)

    b = result[result["filename"] == "b.jpg"].iloc[0]
    assert b["num_total_birds"] == 1
    assert b["num_hummingbirds"] == 1
    assert b["num_other"] == 0


def test_aggregate_to_photos_with_duplicate_flags():
    """Duplicate flags are propagated to photo level."""
    df = pd.DataFrame(
        [
            {
                "filename": "a.jpg",
                "date": "07 APR 2025",
                "time": "06:30 AM",
                "temperature": 72.0,
                "camera": "CAMERA 1",
                "bird_index": 0,
                "detection_confidence": 0.85,
                "classification_confidence": 0.92,
                "is_hummingbird": True,
                "is_duplicate": False,
                "is_suspected_duplicate": True,
            },
            {
                "filename": "a.jpg",
                "date": "07 APR 2025",
                "time": "06:30 AM",
                "temperature": 72.0,
                "camera": "CAMERA 1",
                "bird_index": 1,
                "detection_confidence": 0.75,
                "classification_confidence": 0.45,
                "is_hummingbird": False,
                "is_duplicate": False,
                "is_suspected_duplicate": True,
            },
        ]
    )
    result = aggregate_to_photos(df)
    assert len(result) == 1
    assert result.iloc[0]["is_duplicate"] == False
    assert result.iloc[0]["is_suspected_duplicate"] == True


def test_aggregate_to_photos_empty():
    df = pd.DataFrame(columns=["filename"])
    result = aggregate_to_photos(df)
    assert len(result) == 0
    assert "num_total_birds" in result.columns
