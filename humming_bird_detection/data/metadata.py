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


def parse_metadata(filename: str, margin: np.ndarray) -> PhotoMetadata:
    """Run OCR on a margin strip and extract structured camera metadata."""
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

    temp_raw = _match(r"(\d+)\s*[F'`\"]", "temperature")
    temperature = float(temp_raw) if temp_raw is not None else None

    camera_id = _match(r"(CAMERA\s+\d+)", "camera_id")
    date = _match(r"(\d{1,2}\s+[A-Z]{3}\s+\d{4})", "date")
    time = _match(r"(\d{1,2}\s*:\s*\d{2}\s*[AP]M)", "time")

    return PhotoMetadata(
        filename=filename,
        temperature=temperature,
        camera_id=camera_id,
        date=date,
        time=time,
    )
