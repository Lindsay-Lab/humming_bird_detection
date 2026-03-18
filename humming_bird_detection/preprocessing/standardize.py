#standardize.py
"""Image standardisation: RGB conversion, long-side downscale, save as JPEG.

If the source is already JPEG/JPG and does not need resizing, the file is
copied as-is to avoid recompression quality loss.  Only images that need
resizing (long side > max_long_side) are re-encoded.

No EXIF transpose (EDA confirmed not needed for this dataset).
No offline augmentation — use YOLO's built-in augmentations during training
(hsv_v handles brightness, hsv_s handles contrast/saturation).
Parallelised with ProcessPoolExecutor for speed on large datasets.
Logs before/after pixel statistics per image.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from humming_bird_detection.configs.config import PrepConfig

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
_JPEG_EXTS = frozenset((".jpg", ".jpeg"))

_RESAMPLE_MAP = {
    "LANCZOS": Image.LANCZOS,
    "BICUBIC": Image.BICUBIC,
    "BILINEAR": Image.BILINEAR,
    "NEAREST": Image.NEAREST,
}

def _get_resample(name: str):
    return _RESAMPLE_MAP.get(name.upper(), Image.LANCZOS)

def _pixel_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute pixel statistics for a uint8 RGB image array."""
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean_r": float(arr[:, :, 0].mean()),
        "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
    }

def _fmt_stats(s: Dict[str, float]) -> str:
    return (
        f"mean={s['mean']:.1f} std={s['std']:.1f} "
        f"min={s['min']:.0f} max={s['max']:.0f} "
        f"R={s['mean_r']:.1f} G={s['mean_g']:.1f} B={s['mean_b']:.1f}"
    )

def standardize_image(
    src_img: Path,
    dst_img: Path,
    cfg: PrepConfig,
) -> Optional[Dict]:
    """RGB convert + long-side downscale + save as JPEG.

    If the source is already JPEG and does NOT need resizing, copy the
    file as-is, only re-encode when the image actually needs to be resized.

    Returns dict with before/after stats on success, None on failure.
    """
    try:
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        is_jpeg_src = src_img.suffix.lower() in _JPEG_EXTS

        with Image.open(src_img) as im:
            im = im.convert("RGB")
            before_w, before_h = im.size
            before_arr = np.asarray(im)
            before_stats = _pixel_stats(before_arr)

            long_side = max(before_w, before_h)
            resized = False
            if long_side > cfg.max_long_side:
                scale = cfg.max_long_side / long_side
                new_w = int(round(before_w * scale))
                new_h = int(round(before_h * scale))
                resample = _get_resample(cfg.interpolation)
                im = im.resize((new_w, new_h), resample)
                resized = True

            after_w, after_h = im.size
            after_arr = np.asarray(im)
            after_stats = _pixel_stats(after_arr)

            if resized or not is_jpeg_src:
                im.save(dst_img, format="JPEG", quality=cfg.jpeg_quality)
            else:
                shutil.copy2(src_img, dst_img)

        return {
            "name": src_img.name,
            "before_size": (before_w, before_h),
            "after_size": (after_w, after_h),
            "resized": resized,
            "copied": is_jpeg_src and not resized,
            "before_pixels": before_stats,
            "after_pixels": after_stats,
        }
    except Exception as exc:
        logger.warning(f"FAILED {src_img.name}: {exc}")
        return None

def _worker(args: Tuple) -> Tuple[str, Optional[Dict]]:
    """Top-level picklable worker for ProcessPoolExecutor."""
    src_img, dst_img, src_lbl, dst_lbl, cfg = args
    src_img, dst_img, src_lbl, dst_lbl = Path(src_img), Path(dst_img), Path(src_lbl), Path(dst_lbl)
    info = standardize_image(src_img, dst_img, cfg)
    if info is not None:
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("")
    return src_img.stem, info

def standardize_dataset(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    cfg: PrepConfig,
) -> Tuple[int, int]:
    """Standardize all images in parallel. Logs before/after stats per image.
    Returns (kept_count, skipped_count).
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    src_images = sorted(
        p for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    logger.info(
        f"Standardizing {len(src_images)} images "
        f"(max_long_side={cfg.max_long_side}, interpolation={cfg.interpolation}, "
        f"jpeg_quality={cfg.jpeg_quality}, workers={cfg.num_workers})"
    )

    work_items = []
    for img in src_images:
        out_name = f"{img.stem}{cfg.final_img_ext}"
        dst = dst_img_dir / out_name
        src_lbl = src_lbl_dir / f"{img.stem}.txt"
        dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
        work_items.append((str(img), str(dst), str(src_lbl), str(dst_lbl), cfg))

    kept = 0
    skipped = 0
    resized_count = 0
    # Accumulators for dataset-level stats
    all_before_means: List[float] = []
    all_after_means: List[float] = []

    with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Standardizing", unit="img", dynamic_ncols=True):
            stem, info = fut.result()
            if info is None:
                skipped += 1
                continue
            kept += 1
            if info["resized"]:
                resized_count += 1
            all_before_means.append(info["before_pixels"]["mean"])
            all_after_means.append(info["after_pixels"]["mean"])
            # Per-image log (debug level to avoid flooding console)
            logger.debug(
                f"[{info['name']}] "
                f"size {info['before_size']} -> {info['after_size']} "
                f"resized={info['resized']} | "
                f"BEFORE: {_fmt_stats(info['before_pixels'])} | "
                f"AFTER:  {_fmt_stats(info['after_pixels'])}"
            )

    # Dataset-level summary
    if all_before_means:
        b_mean = np.mean(all_before_means)
        b_std = np.std(all_before_means)
        a_mean = np.mean(all_after_means)
        a_std = np.std(all_after_means)
        logger.info(
            f"Pixel stats across dataset — "
            f"BEFORE: mean_of_means={b_mean:.1f} std_of_means={b_std:.1f} | "
            f"AFTER:  mean_of_means={a_mean:.1f} std_of_means={a_std:.1f}"
        )

    logger.info(
        f"Standardization complete: {kept} kept, {skipped} skipped, "
        f"{resized_count} resized (long side > {cfg.max_long_side})"
    )
    return kept, skipped