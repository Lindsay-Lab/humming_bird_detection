"""Tests for humming_bird_detection.data.preprocessing."""

from __future__ import annotations

import numpy as np

from humming_bird_detection.data.preprocessing import crop_and_resize


def test_output_shape():
    image = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)
    result = crop_and_resize(image, bbox=(50, 50, 200, 200))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_padding_clamped_at_border():
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    # bbox at top-left corner — padding would go out of bounds
    result = crop_and_resize(image, bbox=(0, 0, 50, 50))
    assert result.shape == (224, 224, 3)


def test_large_pad_fraction():
    image = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)
    result = crop_and_resize(image, bbox=(100, 100, 200, 200), pad_fraction=0.5)
    assert result.shape == (224, 224, 3)


def test_non_square_bbox_no_distortion():
    """A tall narrow bbox must be letterboxed, not stretched."""
    # Construct image with a distinct vertical stripe (red column, blue elsewhere)
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[:, :] = [255, 0, 0]  # blue everywhere (BGR)
    image[:, 490:510] = [0, 0, 255]  # red vertical stripe

    # Tall narrow bbox centred on the stripe but wide enough to include blue background
    # on both sides (90% blue, 10% red) so the letterbox fill mean is distinctly blue.
    result = crop_and_resize(image, bbox=(400, 200, 600, 840), pad_fraction=0.0)
    assert result.shape == (224, 224, 3)

    # The red stripe should appear in the centre columns, not stretched to full width.
    # Centre column of output should be reddish; far-left column should be bluish.
    centre_col = result[:, 112, :]  # BGR: expect red → high R (index 2), low B
    left_col = result[:, 10, :]  # letterbox fill — bluish or mean-ish, not red
    assert centre_col[:, 2].mean() > 150, "Centre of output should preserve red stripe"
    assert (
        left_col[:, 2].mean() < centre_col[:, 2].mean()
    ), "Left edge (letterbox fill) should have less red than the stripe centre"


def test_square_bbox_unchanged_path():
    """A square bbox should pass through without any letterboxing overhead."""
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    result = crop_and_resize(image, bbox=(50, 50, 150, 150), pad_fraction=0.0)
    assert result.shape == (224, 224, 3)
