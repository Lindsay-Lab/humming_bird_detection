"""Post-processing steps applied to pipeline output DataFrames.

Each post-processor is a callable with signature::

    (df: pd.DataFrame) -> pd.DataFrame

and must return a new DataFrame (or the same one mutated) with any additional
columns added.  Post-processors are composable: the output of one becomes the
input of the next.

To add a new post-processing step:
1. Implement a function matching the signature above.
2. Register it in :func:`build_postprocessors` if it should run by default when
   the corresponding config section is present.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

PostProcessor = Callable[[pd.DataFrame], pd.DataFrame]

_DATE_TIME_FORMAT = "%d %b %Y %I:%M %p"  # e.g. "07 APR 2025 06:32 AM"


def _parse_datetime(date: str | None, time: str | None) -> datetime | None:
    """Parse separate date and time strings into a datetime.

    Returns None when either field is missing or unparseable.  The resulting
    precision is one minute — the resolution available from OCR metadata.
    """
    if date is None or time is None:
        return None
    try:
        return datetime.strptime(f"{date} {time}", _DATE_TIME_FORMAT)
    except ValueError:
        return None


def flag_duplicates(df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    """Add an ``is_duplicate`` column to a detections DataFrame.

    A detection row is flagged as a duplicate when its image was captured
    within *window_seconds* of the most recent non-duplicate image.  All rows
    belonging to the same image file (same ``filename``) receive the same flag.

    Successive images are ordered by (parsed_datetime, filename); the filename
    acts as a tiebreaker for images that share the same minute-precision
    timestamp.

    Rows whose ``date`` or ``time`` metadata could not be parsed are conservatively
    treated as non-duplicates (``is_duplicate=False``) and do not update the
    reference timestamp — they are excluded from the window comparison.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`pipeline.process_directory`.  Must contain
        columns ``filename``, ``date``, and ``time``.
    window_seconds:
        Maximum gap in seconds between two consecutive images for the later
        one to be considered a duplicate of the earlier sighting.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``is_duplicate`` boolean column.
    """
    if df.empty:
        return df.assign(is_duplicate=pd.Series(dtype=bool))

    # Build a per-filename lookup: (parsed_datetime, filename) → is_duplicate
    # Use the first occurrence of each filename to read its date/time.
    first_rows = df.drop_duplicates(subset="filename")[["filename", "date", "time"]]

    records: list[tuple[datetime | None, str]] = []
    for _, row in first_rows.iterrows():
        dt = _parse_datetime(row["date"], row["time"])
        records.append((dt, row["filename"]))

    # Sort by (datetime, filename); place unparseable timestamps last so they
    # do not erroneously "anchor" the deduplication chain.
    def _sort_key(item: tuple[datetime | None, str]):
        dt, fname = item
        return (dt is None, dt or datetime.min, fname)

    records.sort(key=_sort_key)

    is_dup: dict[str, bool] = {}
    last_reference_dt: datetime | None = None

    for dt, fname in records:
        if dt is None:
            # Cannot determine position in time → not a duplicate, skip chain
            is_dup[fname] = False
            continue

        if last_reference_dt is None:
            is_dup[fname] = False
            last_reference_dt = dt
        elif (dt - last_reference_dt).total_seconds() <= window_seconds:
            is_dup[fname] = True
            # Do NOT update last_reference_dt: all duplicates trace back to
            # the original sighting, so the window is measured from there.
        else:
            is_dup[fname] = False
            last_reference_dt = dt

    result = df.copy()
    result["is_duplicate"] = result["filename"].map(is_dup)
    n_dup = result["is_duplicate"].sum()
    logger.info(
        "Duplicate flagging (window=%.0fs): %d/%d records marked as duplicates",
        window_seconds,
        n_dup,
        len(result),
    )
    return result


def flag_suspected_duplicates(
    df: pd.DataFrame, window_seconds: float = 60
) -> pd.DataFrame:
    """Flag composition-based suspected duplicates within a time window.

    Within each (camera, date, minute) group, photos whose bird-composition
    signature — (num_hummingbirds, num_other) — matches another photo in the
    same minute are considered candidates.  Among N candidates with identical
    composition, only ``ceil(N / 3)`` are kept; the remainder are flagged as
    suspected duplicates.

    The rule encodes: 1 photo → count 1; 2–3 → count 1; 4–6 → count 2;
    7–9 → count 3; and so on (``math.ceil(N / 3)``).

    All bird-level rows belonging to the same filename share the same flag.

    Parameters
    ----------
    df:
        Bird-level DataFrame (one row per bird).  Must contain columns
        ``filename``, ``camera``, ``date``, ``time``, ``is_hummingbird``.
    window_seconds:
        Time window in seconds within which photos are grouped.  Default 60
        groups photos taken in the same clock minute, which is the natural
        resolution of the OCR metadata.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``is_suspected_duplicate`` boolean column.
    """
    import math

    if df.empty:
        return df.assign(is_suspected_duplicate=pd.Series(dtype=bool))

    # --- Build per-photo composition signature ---
    # Count birds per filename, split by class.
    photo_stats = (
        df.groupby("filename")
        .agg(
            n_hummingbird=("is_hummingbird", lambda x: int((x == 1).sum())),
            n_other=("is_hummingbird", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )
    # Compose a string signature so grouping is trivial.
    photo_stats["_comp_sig"] = photo_stats.apply(
        lambda r: f"HB{r['n_hummingbird']}_OT{r['n_other']}", axis=1
    )

    # --- Attach camera, datetime, and minute bucket ---
    first_rows = df.drop_duplicates(subset="filename")[
        ["filename", "camera", "date", "time"]
    ]
    photo_stats = photo_stats.merge(first_rows, on="filename", how="left")

    dt_series = pd.to_datetime(
        photo_stats["date"].fillna("") + " " + photo_stats["time"].fillna(""),
        format=_DATE_TIME_FORMAT,
        errors="coerce",
    )
    photo_stats["_dt"] = dt_series
    photo_stats["_minute_bucket"] = dt_series.dt.floor(f"{window_seconds}s")

    # --- Group by (camera, date, minute bucket, composition signature) ---
    group_keys = ["camera", "_minute_bucket", "_comp_sig"]
    # Fill NA camera so NaN groups don't collapse unexpectedly.
    photo_stats["_cam_fill"] = photo_stats["camera"].fillna("__UNKNOWN__")

    is_suspected: dict[str, bool] = {}

    for (_cam, _bucket, _sig), group in photo_stats.groupby(
        ["_cam_fill", "_minute_bucket", "_comp_sig"], dropna=False
    ):
        n = len(group)
        keep_n = max(1, math.ceil(n / 3))
        # Keep the first `keep_n` photos (sorted by datetime then filename for
        # deterministic tie-breaking); flag the rest.
        sorted_fnames = group.sort_values(
            ["_dt", "filename"], na_position="last"
        )["filename"].tolist()
        for i, fname in enumerate(sorted_fnames):
            is_suspected[fname] = i >= keep_n

    result = df.copy()
    result["is_suspected_duplicate"] = result["filename"].map(is_suspected).fillna(False)
    n_sus = int(result["is_suspected_duplicate"].sum())
    logger.info(
        "Suspected-duplicate flagging (window=%.0fs, ceil(N/3) rule): "
        "%d/%d records flagged",
        window_seconds,
        n_sus,
        len(result),
    )
    return result


def aggregate_to_photos(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bird-level records to one row per photo.

    The returned DataFrame has columns:

    - ``filename``, ``date``, ``time``, ``temperature``, ``camera``
      (taken from the first row of each photo).
    - ``num_total_birds``, ``num_hummingbirds``, ``num_other``
      (counts per photo).
    - ``mean_detection_conf``, ``max_detection_conf``,
      ``mean_classification_conf``, ``max_classification_conf``
      (aggregated confidence values).
    - ``has_hummingbird`` — True if at least one bird in the photo was
      classified as a hummingbird.
    - ``is_duplicate``, ``is_suspected_duplicate`` — propagated from
      bird-level flags when present.

    Parameters
    ----------
    df:
        Bird-level DataFrame (one row per bird).

    Returns
    -------
    pd.DataFrame
        One row per unique filename.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "filename",
                "date",
                "time",
                "temperature",
                "camera",
                "num_total_birds",
                "num_hummingbirds",
                "num_other",
                "mean_detection_conf",
                "max_detection_conf",
                "mean_classification_conf",
                "max_classification_conf",
                "has_hummingbird",
            ]
        )

    # Build named aggregation kwargs for pandas groupby().agg().
    agg_kwargs: dict[str, tuple[str, str | Callable]] = {
        "date": ("date", "first"),
        "time": ("time", "first"),
        "temperature": ("temperature", "first"),
        "camera": ("camera", "first"),
        "num_total_birds": ("bird_index", "count"),
        "num_hummingbirds": ("is_hummingbird", lambda x: int((x == 1).sum())),
        "num_other": ("is_hummingbird", lambda x: int((x == 0).sum())),
        "has_hummingbird": ("is_hummingbird", lambda x: (x == 1).any()),
        "mean_detection_conf": ("detection_confidence", "mean"),
        "max_detection_conf": ("detection_confidence", "max"),
        "mean_classification_conf": ("classification_confidence", "mean"),
        "max_classification_conf": ("classification_confidence", "max"),
    }

    # Propagate duplicate flags if present.
    for col in ["is_duplicate", "is_suspected_duplicate"]:
        if col in df.columns:
            agg_kwargs[col] = (col, "first")

    result = df.groupby("filename", sort=False).agg(**agg_kwargs).reset_index()

    # Ensure integer counts.
    for col in ["num_total_birds", "num_hummingbirds", "num_other"]:
        if col in result.columns:
            result[col] = result[col].astype(int)

    return result


def apply_postprocessors(
    df: pd.DataFrame, processors: list[PostProcessor]
) -> pd.DataFrame:
    """Apply a sequence of post-processors to a DataFrame in order.

    Each processor receives the output of the previous one.
    """
    for proc in processors:
        df = proc(df)
    return df


def build_postprocessors(cfg: dict) -> list[PostProcessor]:
    """Return the list of post-processors implied by *cfg*.

    Currently registered post-processors:

    ``deduplication`` section present
        :func:`flag_duplicates` with ``window_seconds`` from config.

    ``suspected_duplicates`` section present
        :func:`flag_suspected_duplicates` with ``window_seconds`` from config
        (default 60, grouping photos within the same clock minute).
    """
    processors: list[PostProcessor] = []

    if "deduplication" in cfg:
        window = float(cfg["deduplication"]["window_seconds"])
        processors.append(lambda df, w=window: flag_duplicates(df, w))

    if "suspected_duplicates" in cfg:
        window = float(cfg["suspected_duplicates"].get("window_seconds", 60))
        processors.append(lambda df, w=window: flag_suspected_duplicates(df, w))

    return processors
