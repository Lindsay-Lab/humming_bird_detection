#config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PrepConfig:
    """Central configuration for the YOLO preprocessing pipeline.

    Class mapping (after remapping):
        0: hummingbird
        1: other_bird  (e.g. bananaquit)
        Background images have empty label files (no bounding boxes).
    """

    # ── Class mapping
    hummingbird_ids: Optional[set[int]] = None
    hummingbird_name_hints: Tuple[str, ...] = ("humming", "hummingbird")
    class_names: Tuple[str, ...] = ("hummingbird", "other_bird")

    # ── Image standardisation
    max_long_side: int = 1920          # downscale so longest edge <= this
    final_img_ext: str = ".jpg"
    jpeg_quality: int = 95
    interpolation: str = "LANCZOS"     # PIL resampling filter name

    # ── Label validation thresholds
    coord_min: float = 0.0
    coord_max: float = 1.0
    min_box_dim: float = 1e-6          # minimum normalised w or h

    # ── Splitting (positive images only)
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # ── Timestamp cropping
    crop_timestamp: bool = True
    timestamp_crop_fraction: float = 0.0104  # bottom ~1% of image height

    # ── Visualization
    visualize_samples: int = 16              # save this many sample images per split

    # ── Parallelism
    num_workers: int = 8

    # ── Training (passed through to YOLO trainer)
    batch_size: int = 16                     # overridable via CLI --batch-size

    # ── Background separation (pretrained COCO detection model)
    bg_sep_model: str = "yolo26n.pt"
    bg_sep_conf: float = 0.25
    bg_sep_iou: float = 0.45
    bg_sep_bird_classes: Tuple[int, ...] = (14,)  # COCO class 14 = bird
    bg_sep_device: str = ""                        # "" = auto (cuda if available)