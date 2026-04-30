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

    def _match(pattern: str, field: str) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        if m is None:
            logger.warning(
                "Could not parse field '%s' from '%s'. OCR text: %r",
                field,
                filename,
                text,
            )
            return None
        return m.group(1)

    # Temperature: digits followed by optional whitespace and a Fahrenheit marker.
    # The pattern accepts F, single/smart quotes ('), and double/smart quotes (").
    # Only Fahrenheit is supported; the cameras this project targets report in °F.
    temp_raw = _match(r"(\d+)\s*[F'`\"]", "temperature")
    temperature = float(temp_raw) if temp_raw is not None else None

    # Camera ID: literal word CAMERA followed by one or more digits.
    camera_id = _match(r"(CAMERA\s+\d+)", "camera_id")
    # Date: 1–2 digit day, 3-letter uppercase month abbreviation, 4-digit year.
    date = _match(r"(\d{1,2}\s+[A-Z]{3}\s+\d{4})", "date")
    # Time: 12-hour format with AM/PM; tolerates missing space before meridiem.
    time = _match(r"(\d{1,2}\s*:\s*\d{2}\s*[AP]M)", "time")

    return PhotoMetadata(
        filename=filename,
        temperature=temperature,
        camera_id=camera_id,
        date=date,
        time=time,
    )
