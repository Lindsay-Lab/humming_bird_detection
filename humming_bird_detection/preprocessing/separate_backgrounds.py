#separate_backgrounds.py
"""Separate background (no-bird) images from positive images using a
pretrained YOLO model (COCO-trained) to detect birds.

Uses YOLO26n pretrained on COCO (class 14 = bird). Images where no bird
is detected above the confidence threshold are classified as backgrounds
and moved to a separate directory.

This is a preprocessing step, only inference.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from humming_bird_detection.configs.config import PrepConfig

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def _resolve_device(cfg: PrepConfig) -> str:
    if cfg.bg_sep_device:
        return cfg.bg_sep_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def separate_backgrounds(
    img_dir: Path,
    lbl_dir: Path,
    positive_img_dir: Path,
    positive_lbl_dir: Path,
    background_img_dir: Path,
    background_lbl_dir: Path,
    cfg: PrepConfig,
) -> Dict[str, int]:
    """Run pretrained YOLO inference to split images into positive / background.

    Images where the pretrained model detects at least one bird (COCO class 14)
    above cfg.bg_sep_conf are treated as positive. The rest go to backgrounds.

    Parameters
    ----------
    img_dir : Path
        Source images directory.
    lbl_dir : Path
        Source labels directory (labels are moved alongside their images).
    positive_img_dir / positive_lbl_dir : Path
        Destination for images that contain birds.
    background_img_dir / background_lbl_dir : Path
        Destination for images with no detected birds.
    cfg : PrepConfig
        Configuration (model name, confidence, bird classes, device).

    Returns dict with counts: {"positive": N, "background": M, "failed": F}.
    """
    from ultralytics import YOLO

    device = _resolve_device(cfg)
    logger.info(f"Loading pretrained model: {cfg.bg_sep_model} on {device}")
    model = YOLO(cfg.bg_sep_model)

    bird_classes = set(cfg.bg_sep_bird_classes)

    for d in (positive_img_dir, positive_lbl_dir, background_img_dir, background_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    logger.info(
        f"Running background separation on {len(images)} images "
        f"(conf={cfg.bg_sep_conf}, bird_classes={sorted(bird_classes)}, device={device})"
    )

    counts = {"positive": 0, "background": 0, "failed": 0}

    # Batch inference for speed
    batch_size = 32
    for start in tqdm(range(0, len(images), batch_size), desc="BG separation", unit="batch", dynamic_ncols=True):
        batch_paths = images[start : start + batch_size]
        try:
            results = model.predict(
                source=[str(p) for p in batch_paths],
                conf=cfg.bg_sep_conf,
                iou=cfg.bg_sep_iou,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning(f"Batch inference failed (batch {start}): {exc}")
            counts["failed"] += len(batch_paths)
            continue

        for img_path, result in zip(batch_paths, results):
            # Check if any detected class is a bird
            detected_classes = set()
            if result.boxes is not None and len(result.boxes) > 0:
                detected_classes = set(result.boxes.cls.int().tolist())

            has_bird = bool(detected_classes & bird_classes)

            if has_bird:
                dst_img_dir = positive_img_dir
                dst_lbl_dir = positive_lbl_dir
                counts["positive"] += 1
            else:
                dst_img_dir = background_img_dir
                dst_lbl_dir = background_lbl_dir
                counts["background"] += 1

            # Move image
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            # Move label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
            else:
                (dst_lbl_dir / f"{img_path.stem}.txt").write_text("")

    logger.info(
        f"Background separation complete: "
        f"positive={counts['positive']}, background={counts['background']}, "
        f"failed={counts['failed']}"
    )
    return counts


def main():
    """CLI entry point for standalone background separation.

    When run standalone (not via the preprocessor), this expects a flat
    images/ + labels/ directory as input.  The preprocessor calls
    separate_backgrounds() directly with its own interim paths.
    """
    from humming_bird_detection.configs.paths import LOGS_DIR, PROCESSED_DATA_DIR
    from humming_bird_detection.utils.logging import setup_logging

    setup_logging(LOGS_DIR, "bg_sep", "INFO")
    cfg = PrepConfig()

    # Expect standardized images in the processed interim dir
    base = PROCESSED_DATA_DIR / "hb_detection" / "standardized"
    img_dir = base / "images"
    lbl_dir = base / "labels"

    out_base = PROCESSED_DATA_DIR / "hb_detection"
    positive_img = out_base / "positive" / "images"
    positive_lbl = out_base / "positive" / "labels"
    background_img = out_base / "backgrounds" / "images"
    background_lbl = out_base / "backgrounds" / "labels"

    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        logger.info("Run the preprocessor first, or provide a standardized images dir.")
        return

    counts = separate_backgrounds(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        positive_img_dir=positive_img,
        positive_lbl_dir=positive_lbl,
        background_img_dir=background_img,
        background_lbl_dir=background_lbl,
        cfg=cfg,
    )
    logger.success(f"Done. Results: {counts}")
    logger.info(f"Positive images: {positive_img}")
    logger.info(f"Background images: {background_img}")

if __name__ == "__main__":
    main()
