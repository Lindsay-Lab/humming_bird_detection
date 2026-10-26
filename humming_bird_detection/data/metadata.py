"""OCR-based extraction of camera metadata from photo margin strips."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhotoMetadata:
    filename: str
    temperature: float | None  # degrees F
    camera_id: str | None  # e.g. "CAMERA 3"
    date: str | None  # "DD MON YYYY"  e.g. "07 APR 2025"
    time: str | None  # "HH:MM AM/PM"  e.g. "06:32 AM"


def extract_margin(image: np.ndarray, margin_px: int) -> np.ndarray:
    """Return the bottom margin_px rows of a BGR image array."""
    return image[-margin_px:, :, :].copy()


# Lazy-initialise EasyOCR reader to avoid paying startup cost at import time.
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # noqa: PLC0415

        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def parse_metadata(filename: str, margin: np.ndarray, reader=None) -> PhotoMetadata:
    """Run OCR on a margin strip and extract structured camera metadata.

    Parameters
    ----------
    filename:
        Image filename used only for warning messages.
    margin:
        BGR image array of the bottom metadata strip (output of extract_margin).
    reader:
        Optional pre-constructed easyocr.Reader instance.  When None the
        module-level lazy singleton is used.  Passing an explicit reader is
        useful for testing (avoid loading GPU/CPU OCR) and for controlling
        which language models are loaded.

    Note: EasyOCR inference is non-deterministic on GPU.  For reproducible
    metadata extraction run with gpu=False (the default singleton does this).
    """
    if reader is None:
        reader = _get_reader()
    results = reader.readtext(margin, detail=0)
    text = " ".join(results)

    def _match(pattern: str, field: str, search_text: str | None = None) -> str | None:
        m = re.search(pattern, search_text if search_text is not None else text, re.IGNORECASE)
        if m is None:
            logger.warning(
                "Could not parse field '%s' from '%s'. OCR text: %r",
                field,
                filename,
                text,
            )
            return None
        return m.group(1)

    # --- Temperature ---
    # The cameras this project targets print the ambient temperature in °F
    # immediately before the "CAMERA <n>" label, e.g. "72°F  CAMERA 3".
    # EasyOCR frequently misreads the "°F" glyph cluster as one of %, P, F,
    # o, or a stray quote — real captures include "107 %", "100 P", "0 82 %".
    # We therefore accept any of those trailing markers.  The search is
    # restricted to the text *before* the CAMERA token so that the two-digit
    # minute portion of the timestamp (which is also followed by "PM"/"P")
    # cannot be mistaken for a temperature reading.
    temp_head = re.split(r"CAMERA", text, maxsplit=1, flags=re.IGNORECASE)[0]
    temp_marker = r"(\d{1,3})\s*[FfPp%'`\"°o]"
    temp_raw = _match(temp_marker, "temperature", search_text=temp_head or text)
    temperature = float(temp_raw) if temp_raw is not None else None

    # Camera ID: literal word CAMERA followed by one or more digits.
    camera_id = _match(r"(CAMERA\s+\d+)", "camera_id")
    # Date: 1–2 digit day, 3-letter uppercase month abbreviation, 4-digit year.
    date = _match(r"(\d{1,2}\s+[A-Z]{3}\s+\d{4})", "date")
    # Time: 12-hour clock with AM/PM.  OCR spacing around the colon and before
    # the meridiem is unreliable (e.g. "02 : 28PM", "10 :55AM"), so we capture
    # loosely and then normalise to a canonical "HH:MM AM/PM" string that
    # downstream ``strptime`` parsing (deduplication, hourly/daily activity)
    # can consume.
    time_raw = _match(r"(\d{1,2}\s*:\s*\d{2}\s*[AP]M)", "time")
    time = _normalize_time(time_raw)

    return PhotoMetadata(
        filename=filename,
        temperature=temperature,
        camera_id=camera_id,
        date=date,
        time=time,
    )


def _normalize_time(raw: str | None) -> str | None:
    """Normalise an OCR time string to canonical ``HH:MM AM/PM`` form.

    EasyOCR emits inconsistent spacing around the colon and before the
    meridiem (``02 : 28PM``, ``10 :55AM``, ``6:5 AM``).  We strip all internal
    whitespace, then re-emit a zero-padded 12-hour time with a single space
    before AM/PM so that ``datetime.strptime(..., "%d %b %Y %I:%M %p")``
    succeeds.  Returns None when *raw* is None or cannot be interpreted.
    """
    if raw is None:
        return None
    compact = re.sub(r"\s+", "", raw).upper()
    m = re.match(r"(\d{1,2}):(\d{2})([AP]M)", compact)
    if m is None:
        return None
    hour = int(m.group(1))
    return f"{hour:02d}:{m.group(2)} {m.group(3)}"
