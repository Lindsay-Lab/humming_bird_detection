#extract_timestamp_text.py
"""Lightweight OCR extraction from cropped timestamp strips.

Uses Tesseract OCR (via pytesseract) to read text burned into the bottom
bar of Wingscapes camera images.  Typical text contains:
    - Camera brand / model
    - Temperature
    - Camera ID
    - Date and time (e.g. "03/15/2024 02:32 PM")

This module is intentionally lightweight:
    - CPU-only, no GPU required
    - ~10ms per strip image
    - Only dependency: pytesseract + system Tesseract binary

Usage in the pipeline:
    After crop_timestamp_dataset() saves strips to timestamp_strips/,
    call extract_all_timestamps() to OCR every strip and return a dict
    mapping filename -> parsed metadata.  The preprocessor writes this
    into image_metadata.csv.

HPC install:
    pip install pytesseract
    # Plus system Tesseract:
    #   Ubuntu/Debian: sudo apt-get install tesseract-ocr
    #   Conda:         conda install -c conda-forge tesseract
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def _try_import_pytesseract():
    """Import pytesseract lazily so the rest of the pipeline works without it."""
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        return None


def _preprocess_strip(strip_path: Path):
    """Load strip image and apply light preprocessing for better OCR.
    Returns a PIL Image ready for Tesseract, or None on failure.
    """
    try:
        from PIL import Image, ImageFilter, ImageOps
        with Image.open(strip_path) as im:
            im = im.convert("L")  # grayscale
            im = ImageOps.autocontrast(im)  # maximize contrast
            # Upscale small strips for better OCR accuracy
            w, h = im.size
            if h < 100:
                scale = max(2, 100 // h)
                im = im.resize((w * scale, h * scale), Image.LANCZOS)
            return im.copy()
    except Exception as exc:
        logger.debug(f"Strip preprocessing failed for {strip_path.name}: {exc}")
        return None


def _parse_timestamp_text(raw_text: str) -> Dict[str, str]:
    """Parse raw OCR text into structured fields.

    Attempts to extract:
        - datetime: date and time string
        - temperature: temperature reading (e.g. "78F" or "25C")
        - camera_id: camera identifier if present
        - raw_text: the full OCR output (always present)

    Returns dict with whatever fields could be parsed.
    """
    result: Dict[str, str] = {"raw_text": raw_text.strip()}

    # Clean up common OCR artifacts
    text = raw_text.replace("\n", " ").strip()

    # Try to find date patterns: MM/DD/YYYY, YYYY-MM-DD, DD-MM-YYYY, etc.
    date_patterns = [
        # MM/DD/YYYY HH:MM AM/PM
        r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*[AaPp][Mm])',
        # MM/DD/YYYY HH:MM:SS
        r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2})',
        # MM/DD/YYYY HH:MM
        r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2})',
        # YYYY-MM-DD HH:MM:SS
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
        # YYYY-MM-DD HH:MM
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})',
        # Just a date MM/DD/YYYY
        r'(\d{1,2}/\d{1,2}/\d{4})',
    ]
    for pat in date_patterns:
        match = re.search(pat, text)
        if match:
            result["datetime"] = match.group(1).strip()
            break

    # Temperature: look for digits followed by F or C (with optional degree sign)
    temp_match = re.search(r'(\d{1,3})\s*[°]?\s*([FfCc])\b', text)
    if temp_match:
        result["temperature"] = f"{temp_match.group(1)}{temp_match.group(2).upper()}"

    # Camera ID: look for patterns like "CAM-01", "ID:123", or alphanumeric codes
    cam_match = re.search(r'(?:CAM|ID|Camera)\s*[-:]?\s*(\w+)', text, re.IGNORECASE)
    if cam_match:
        result["camera_id"] = cam_match.group(1)

    return result


def extract_single_timestamp(
    strip_path: Path,
    pytesseract_module,
) -> Dict[str, str]:
    """OCR a single timestamp strip image.
    Returns parsed metadata dict (always contains 'raw_text' key).
    """
    im = _preprocess_strip(strip_path)
    if im is None:
        return {"raw_text": "", "ocr_error": "preprocessing_failed"}

    try:
        # Use --psm 7 (treat as single line) for strip images
        raw_text = pytesseract_module.image_to_string(
            im, config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/:.- °FCAMPamp"
        )
        return _parse_timestamp_text(raw_text)
    except Exception as exc:
        logger.debug(f"OCR failed for {strip_path.name}: {exc}")
        return {"raw_text": "", "ocr_error": str(exc)}


def extract_all_timestamps(
    strip_dir: Path,
) -> Dict[str, Dict[str, str]]:
    """OCR all timestamp strips in a directory.

    Returns dict mapping image filename -> parsed metadata dict.
    If pytesseract is not installed, logs a warning and returns empty dict.
    """
    pytesseract = _try_import_pytesseract()
    if pytesseract is None:
        logger.warning(
            "pytesseract not installed — skipping timestamp OCR extraction. "
            "Install with: pip install pytesseract  (+ system Tesseract binary)"
        )
        return {}

    if not strip_dir.exists():
        logger.warning(f"Timestamp strip directory not found: {strip_dir}")
        return {}

    strips = sorted(
        p for p in strip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    if not strips:
        logger.warning(f"No timestamp strips found in {strip_dir}")
        return {}

    logger.info(f"Extracting text from {len(strips)} timestamp strips (Tesseract OCR)")

    results: Dict[str, Dict[str, str]] = {}
    success = 0
    failed = 0

    for strip_path in strips:
        meta = extract_single_timestamp(strip_path, pytesseract)
        results[strip_path.name] = meta
        if meta.get("raw_text"):
            success += 1
        else:
            failed += 1

    logger.info(
        f"Timestamp OCR complete: {success} extracted, {failed} failed/empty "
        f"out of {len(strips)} strips"
    )

    # Log a few samples
    sample_count = 0
    for fname, meta in results.items():
        if meta.get("datetime") and sample_count < 3:
            logger.info(f"  Sample: {fname} -> {meta}")
            sample_count += 1

    return results
