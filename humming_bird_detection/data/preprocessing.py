"""Crop, pad, and resize image patches around detections."""

from __future__ import annotations

import cv2
import numpy as np


def crop_and_resize(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_fraction: float = 0.10,
    output_size: int = 224,
) -> np.ndarray:
    """Crop a padded region around bbox and resize to (output_size, output_size)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_fraction)
    py = int(bh * pad_fraction)

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)

    crop = image[cy1:cy2, cx1:cx2]
    # INTER_AREA averages pixels in the source region mapped to each output pixel,
    # which produces less aliasing than INTER_LINEAR when downscaling — the most
    # common case here since detections are usually larger than 224 px.
    resized = cv2.resize(
        crop, (output_size, output_size), interpolation=cv2.INTER_AREA
    )
    return resized.astype(np.uint8)
