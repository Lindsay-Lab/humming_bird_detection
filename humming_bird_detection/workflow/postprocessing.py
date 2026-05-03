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
    """
    processors: list[PostProcessor] = []

    if "deduplication" in cfg:
        window = float(cfg["deduplication"]["window_seconds"])
        processors.append(lambda df, w=window: flag_duplicates(df, w))

    return processors
