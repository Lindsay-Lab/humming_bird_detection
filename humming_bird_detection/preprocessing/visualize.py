#visualize.py
"""Save sample preprocessed images with bounding boxes drawn for visual inspection.
Generates sample images from each split or arbitrary dirs so you can verify preprocessing quality.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from PIL import Image, ImageDraw

from humming_bird_detection.configs.config import PrepConfig

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

# Colours for class IDs
_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Default class names if cfg not provided
_DEFAULT_CLASS_NAMES: Dict[int, str] = {0: "hummingbird", 1: "other_bird"}


def _read_yolo_labels(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO label file, return list of (cls, xc, yc, w, h)."""
    if not lbl_path.exists():
        return []
    boxes = []
    with open(lbl_path) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls_id, xc, yc, w, h))
            except (ValueError, TypeError):
                continue
    return boxes


def _draw_boxes(
    im: Image.Image,
    boxes: List[Tuple[int, float, float, float, float]],
    class_names: Optional[Dict[int, str]] = None,
) -> Image.Image:
    """Draw YOLO bounding boxes on image. Returns annotated copy."""
    im = im.copy()
    draw = ImageDraw.Draw(im)
    img_w, img_h = im.size
    names = class_names or _DEFAULT_CLASS_NAMES

    for cls_id, xc, yc, w, h in boxes:
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        color = _COLORS[cls_id % len(_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = names.get(cls_id, f"cls_{cls_id}")
        draw.text((x1, max(0, y1 - 15)), label, fill=color)

    return im

def visualize_dir(
    img_dir: Path,
    lbl_dir: Path,
    output_dir: Path,
    tag: str,
    n_samples: int = 8,
    seed: int = 42,
    class_names: Optional[Dict[int, str]] = None,
) -> int:
    """Save n_samples annotated images from arbitrary images/labels dirs.
    *tag* is used for the output subfolder and log messages.
    Returns number of images saved.
    """
    if not img_dir.exists():
        logger.warning(f"[{tag}] images dir not found: {img_dir}")
        return 0

    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    if not images:
        logger.warning(f"[{tag}] no images found")
        return 0

    rng = random.Random(seed)
    samples = rng.sample(images, min(n_samples, len(images)))

    out_dir = output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img_path in samples:
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                boxes = _read_yolo_labels(lbl_path)
                annotated = _draw_boxes(im, boxes, class_names)

                out_path = out_dir / f"sample_{img_path.name}"
                annotated.save(out_path, format="JPEG", quality=90)
                saved += 1

                logger.debug(
                    f"[{tag}] Saved {out_path.name} "
                    f"(size={im.size}, boxes={len(boxes)})"
                )
        except Exception as exc:
            logger.warning(f"[{tag}] Failed to visualize {img_path.name}: {exc}")

    logger.info(f"[{tag}] Saved {saved} sample visualizations -> {out_dir}")
    return saved


def visualize_split(
    split_dir: Path,
    output_dir: Path,
    split_name: str,
    n_samples: int = 8,
    seed: int = 42,
    class_names: Optional[Dict[int, str]] = None,
) -> int:
    """Save n_samples annotated images from a split (images/ + labels/ subdirs)."""
    return visualize_dir(
        img_dir=split_dir / "images",
        lbl_dir=split_dir / "labels",
        output_dir=output_dir,
        tag=split_name,
        n_samples=n_samples,
        seed=seed,
        class_names=class_names,
    )


def visualize_preprocessed(
    final_root: Path,
    output_dir: Path,
    cfg: PrepConfig,
) -> int:
    """Visualize sample images from all splits in the final dataset.
    Returns total number of sample images saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {i: n for i, n in enumerate(cfg.class_names)}
    total = 0

    for split_name in ("train", "val", "test"):
        split_dir = final_root / split_name
        if split_dir.exists():
            n = visualize_split(
                split_dir=split_dir,
                output_dir=output_dir,
                split_name=split_name,
                n_samples=cfg.visualize_samples,
                seed=cfg.seed,
                class_names=names,
            )
            total += n

    logger.info(f"Total sample visualizations saved: {total} -> {output_dir}")
    return total
