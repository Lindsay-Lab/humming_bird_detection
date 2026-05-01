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
    """Crop a padded region around bbox and resize to (output_size, output_size).

    The crop is letterboxed to a square before resizing so that the aspect
    ratio of the detected region is preserved.  Without this step, a tall
    narrow detection bbox (e.g. 208×640 px from a partial SAHI tile hit) would
    be stretched ~3× horizontally, producing a smeared image that the
    classifier never saw during training.

    Letterbox padding uses the per-channel mean colour of the crop, which is
    less likely to create a hard edge artefact than a constant black fill.
    """
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

    # Letterbox to square so the shorter dimension is centred with neutral fill.
    ch, cw = crop.shape[:2]
    if ch != cw:
        side = max(ch, cw)
        pad_color = np.mean(crop, axis=(0, 1), keepdims=False).astype(np.uint8)
        square = np.full((side, side, 3), pad_color, dtype=np.uint8)
        y_off = (side - ch) // 2
        x_off = (side - cw) // 2
        square[y_off : y_off + ch, x_off : x_off + cw] = crop
        crop = square

    # INTER_AREA averages pixels in the source region mapped to each output pixel,
    # which produces less aliasing than INTER_LINEAR when downscaling — the most
    # common case here since detections are usually larger than 224 px.
    resized = cv2.resize(
        crop, (output_size, output_size), interpolation=cv2.INTER_AREA
    )
    return resized.astype(np.uint8)
