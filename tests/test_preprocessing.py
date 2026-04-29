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
