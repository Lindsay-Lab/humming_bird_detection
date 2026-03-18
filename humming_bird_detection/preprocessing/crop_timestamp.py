#crop_timestamp.py
"""Crop the camera timestamp bar from the bottom of birdfeeder images.

The Wingscapes camera burns a fixed-height info bar at the bottom of every
frame (brand, temperature, camera ID, date/time).

The timestamp strip is saved separately for reference (which bird at what time).
Bounding-box labels are adjusted: any box that extends into the cropped region
is clipped, and boxes that fall entirely inside the cropped region are dropped.

Parallelised with ProcessPoolExecutor.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from PIL import Image
from tqdm import tqdm

from humming_bird_detection.configs.config import PrepConfig

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def _adjust_labels(
    src_lbl: Path,
    dst_lbl: Path,
    crop_frac: float,
) -> Dict[str, int]:
    """Adjust YOLO labels after cropping the bottom crop_frac of the image.

    YOLO normalised coords are relative to image height.  After cropping the
    bottom crop_frac, the new image height = (1 - crop_frac) * old_height.
    We must:
      1. Drop boxes whose centre falls inside the cropped region (yc > 1 - crop_frac).
      2. Clip boxes that partially overlap the crop boundary.
      3. Re-normalise all coords to the new height.

    Returns {"kept": N, "dropped": M, "clipped": C}.
    """
    stats = {"kept": 0, "dropped": 0, "clipped": 0}
    if not src_lbl.exists():
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("")
        return stats

    new_h_frac = 1.0 - crop_frac  # new image height as fraction of old
    lines_out = []

    with open(src_lbl) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, TypeError):
                continue

            # Box boundaries in old normalised coords
            y_top = yc - h / 2
            y_bot = yc + h / 2

            # Drop if centre is in the cropped zone
            if yc >= new_h_frac:
                stats["dropped"] += 1
                continue

            # Clip bottom edge if it extends into crop zone
            clipped = False
            if y_bot > new_h_frac:
                y_bot = new_h_frac
                clipped = True

            # Re-normalise to new image height
            new_yc = ((y_top + y_bot) / 2) / new_h_frac
            new_h = (y_bot - y_top) / new_h_frac
            new_xc = xc  # x is unchanged (no horizontal crop)
            new_w = w     # width unchanged

            # Validate
            if new_h < 1e-6 or new_w < 1e-6:
                stats["dropped"] += 1
                continue

            if clipped:
                stats["clipped"] += 1
            stats["kept"] += 1
            lines_out.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.write_text("\n".join(lines_out) + "\n" if lines_out else "")
    return stats


def crop_single_image(
    src_img: Path,
    dst_img: Path,
    src_lbl: Path,
    dst_lbl: Path,
    timestamp_strip_dir: Path,
    cfg: PrepConfig,
) -> Optional[Dict]:
    """Crop timestamp bar from one image, save strip, adjust labels.
    Returns info dict on success, None on failure.
    """
    try:
        with Image.open(src_img) as im:
            im = im.convert("RGB")
            w, h = im.size
            crop_px = int(round(h * cfg.timestamp_crop_fraction))
            new_h = h - crop_px

            if crop_px <= 0 or new_h <= 0:
                # Nothing to crop — just copy
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
                if src_lbl.exists():
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_lbl, dst_lbl)
                return {
                    "name": src_img.name,
                    "original_size": (w, h),
                    "cropped_size": (w, h),
                    "crop_px": 0,
                    "labels": {"kept": 0, "dropped": 0, "clipped": 0},
                }

            # Crop: keep top portion, discard bottom crop_px rows
            cropped = im.crop((0, 0, w, new_h))
            strip = im.crop((0, new_h, w, h))

            dst_img.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(dst_img, format="JPEG", quality=cfg.jpeg_quality)

            # Save timestamp strip
            timestamp_strip_dir.mkdir(parents=True, exist_ok=True)
            strip.save(timestamp_strip_dir / src_img.name, format="JPEG", quality=cfg.jpeg_quality)

        # Adjust labels
        lbl_stats = _adjust_labels(src_lbl, dst_lbl, cfg.timestamp_crop_fraction)

        return {
            "name": src_img.name,
            "original_size": (w, h),
            "cropped_size": (w, new_h),
            "crop_px": crop_px,
            "labels": lbl_stats,
        }
    except Exception as exc:
        logger.warning(f"FAILED cropping {src_img.name}: {exc}")
        return None


def _crop_worker(args: Tuple) -> Tuple[str, Optional[Dict]]:
    """Picklable worker for ProcessPoolExecutor."""
    src_img, dst_img, src_lbl, dst_lbl, strip_dir, cfg = args
    src_img = Path(src_img)
    dst_img = Path(dst_img)
    src_lbl = Path(src_lbl)
    dst_lbl = Path(dst_lbl)
    strip_dir = Path(strip_dir)
    info = crop_single_image(src_img, dst_img, src_lbl, dst_lbl, strip_dir, cfg)
    return src_img.stem, info


def crop_timestamp_dataset(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    timestamp_strip_dir: Path,
    cfg: PrepConfig,
) -> Tuple[int, int]:
    """Crop timestamp bars from all images in parallel.
    Returns (cropped_count, failed_count).
    """
    src_images = sorted(
        p for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    if not src_images:
        logger.warning(f"No images found in {src_img_dir}")
        return 0, 0

    logger.info(
        f"Cropping timestamp bars from {len(src_images)} images "
        f"(crop_fraction={cfg.timestamp_crop_fraction:.4f}, workers={cfg.num_workers})"
    )

    work_items = []
    for img in src_images:
        dst = dst_img_dir / img.name
        src_lbl = src_lbl_dir / f"{img.stem}.txt"
        dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
        work_items.append((
            str(img), str(dst), str(src_lbl), str(dst_lbl),
            str(timestamp_strip_dir), cfg,
        ))

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    cropped = 0
    failed = 0
    total_lbl_kept = 0
    total_lbl_dropped = 0
    total_lbl_clipped = 0

    with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
        futures = {pool.submit(_crop_worker, item): item[0] for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Cropping timestamps", unit="img", dynamic_ncols=True):
            stem, info = fut.result()
            if info is None:
                failed += 1
                continue
            cropped += 1
            total_lbl_kept += info["labels"]["kept"]
            total_lbl_dropped += info["labels"]["dropped"]
            total_lbl_clipped += info["labels"]["clipped"]
            logger.debug(
                f"[{info['name']}] {info['original_size']} -> {info['cropped_size']} "
                f"(cropped {info['crop_px']}px) | "
                f"labels: kept={info['labels']['kept']} "
                f"dropped={info['labels']['dropped']} clipped={info['labels']['clipped']}"
            )

    logger.info(
        f"Timestamp cropping complete: {cropped} cropped, {failed} failed | "
        f"Labels: {total_lbl_kept} kept, {total_lbl_clipped} clipped, {total_lbl_dropped} dropped"
    )
    logger.info(f"Timestamp strips saved to: {timestamp_strip_dir}")
    return cropped, failed
