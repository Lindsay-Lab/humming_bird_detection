#validate.py
"""Label sanitisation and dataset validation.

1. sanitize_label_file read a single YOLO label, drop rows with
   invalid coordinates (w≤0, h≤0, out-of-range centres), rewrite in place.
2. sanitize_labels bulk-sanitize every label under a directory tree.
3. validate_dataset post-split checks: every image has a label,
   every label has valid rows, no zero-positive splits.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from humming_bird_detection.configs.config import PrepConfig

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

def read_yolo_lines(lbl: Path) -> List[List[float]]:
    """Parse a YOLO label file into [[class, xc, yc, w, h], …]."""
    if not lbl.exists():
        return []
    rows: List[List[float]] = []
    with open(lbl, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                c = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                rows.append([float(c), xc, yc, w, h])
            except (ValueError, TypeError):
                continue
    return rows

def write_yolo_lines(lbl: Path, rows: List[List[float]]) -> None:
    """Write YOLO rows back to *lbl*, creating parent dirs if needed."""
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        c = int(r[0])
        xc, yc, w, h = r[1], r[2], r[3], r[4]
        lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    lbl.write_text("\n".join(lines) + "\n" if lines else "")

@dataclass
class SanitizeResult:
    path: Path
    original_rows: int = 0
    kept_rows: int = 0
    dropped_rows: int = 0
    issues: List[str] = field(default_factory=list)

def sanitize_label_file(
    lbl: Path,
    cfg: PrepConfig,
    *,
    fix_in_place: bool = True,
) -> SanitizeResult:
    """Validate and optionally rewrite a single YOLO label file.

    Drops rows where:
    - w ≤ min_box_dim  or  h ≤ min_box_dim
    - xc, yc, w, h outside [coord_min, coord_max]
    - class id < 0
    """
    result = SanitizeResult(path=lbl)
    rows = read_yolo_lines(lbl)
    result.original_rows = len(rows)

    clean: List[List[float]] = []
    for r in rows:
        c, xc, yc, w, h = int(r[0]), r[1], r[2], r[3], r[4]
        issues: List[str] = []

        if c < 0:
            issues.append(f"negative class_id={c}")
        if w <= cfg.min_box_dim or h <= cfg.min_box_dim:
            issues.append(f"non-positive dims w={w:.6f} h={h:.6f}")
        if not (cfg.coord_min <= xc <= cfg.coord_max):
            issues.append(f"xc={xc:.6f} out of range")
        if not (cfg.coord_min <= yc <= cfg.coord_max):
            issues.append(f"yc={yc:.6f} out of range")
        if not (cfg.coord_min <= w <= cfg.coord_max):
            issues.append(f"w={w:.6f} out of range")
        if not (cfg.coord_min <= h <= cfg.coord_max):
            issues.append(f"h={h:.6f} out of range")

        if issues:
            result.issues.extend(
                [f"row ({c} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}): {i}" for i in issues]
            )
        else:
            clean.append(r)

    result.kept_rows = len(clean)
    result.dropped_rows = result.original_rows - result.kept_rows

    if fix_in_place and result.dropped_rows > 0:
        write_yolo_lines(lbl, clean)
        logger.info(
            f"Sanitized {lbl.name}: kept {result.kept_rows}/{result.original_rows} rows "
            f"(dropped {result.dropped_rows})"
        )

    return result

def sanitize_labels(
    label_dir: Path,
    cfg: PrepConfig,
    *,
    fix_in_place: bool = True,
) -> List[SanitizeResult]:
    """Sanitize every .txt label under *label_dir* (non-recursive, parallel).

    Returns list of per-file results.
    """
    if not label_dir.exists():
        logger.warning(f"Label directory does not exist: {label_dir}")
        return []

    label_files = sorted(label_dir.glob("*.txt"))
    logger.info(f"Sanitizing {len(label_files)} label files ({cfg.num_workers} workers)")

    results: List[SanitizeResult] = []
    total_dropped = 0
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as pool:
        futures = {
            pool.submit(sanitize_label_file, lbl, cfg, fix_in_place=fix_in_place): lbl
            for lbl in label_files
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            total_dropped += r.dropped_rows

    n_affected = sum(1 for r in results if r.dropped_rows > 0)
    logger.info(
        f"Label sanitisation complete: {n_affected} files modified, "
        f"{total_dropped} total rows dropped across {len(results)} files"
    )
    return results

@dataclass
class SplitStats:
    split: str
    total_images: int = 0
    total_labels: int = 0
    positive_images: int = 0
    background_images: int = 0
    total_boxes: int = 0
    invalid_rows: int = 0
    missing_labels: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)

def validate_split(
    split_root: Path,
    split: str,
    cfg: PrepConfig,
) -> SplitStats:
    """Validate a single split directory (split_root/split/images + labels)."""
    stats = SplitStats(split=split)
    img_dir = split_root / split / "images"
    lbl_dir = split_root / split / "labels"

    if not img_dir.exists():
        logger.warning(f"[{split}] images dir missing: {img_dir}")
        return stats

    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    stats.total_images = len(imgs)

    for img in imgs:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            stats.missing_labels += 1
            continue

        stats.total_labels += 1
        rows = read_yolo_lines(lbl)

        if not rows:
            stats.background_images += 1
            continue

        stats.positive_images += 1
        for r in rows:
            c, xc, yc, w, h = int(r[0]), r[1], r[2], r[3], r[4]
            if w <= cfg.min_box_dim or h <= cfg.min_box_dim:
                stats.invalid_rows += 1
            else:
                stats.total_boxes += 1
                stats.class_counts[c] = stats.class_counts.get(c, 0) + 1

    return stats

def validate_dataset(
    split_root: Path,
    cfg: PrepConfig,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> List[SplitStats]:
    """Run post-split validation and assert critical invariants.

    Raises RuntimeError if:
    - val or test has 0 positive images
    - any split has invalid label rows remaining
    - images are missing label files
    """
    logger.info(f"Validating dataset at {split_root}")
    all_stats: List[SplitStats] = []
    errors: List[str] = []

    for split in splits:
        s = validate_split(split_root, split, cfg)
        all_stats.append(s)

        logger.info(
            f"[{split}] images={s.total_images} positive={s.positive_images} "
            f"background={s.background_images} boxes={s.total_boxes} "
            f"invalid_rows={s.invalid_rows} missing_labels={s.missing_labels}"
        )
        if s.class_counts:
            dist = ", ".join(f"c{k}={v}" for k, v in sorted(s.class_counts.items()))
            logger.info(f"[{split}] class distribution: {dist}")

        if s.invalid_rows > 0:
            errors.append(f"[{split}] has {s.invalid_rows} invalid label rows remaining")
        if s.missing_labels > 0:
            errors.append(f"[{split}] has {s.missing_labels} images without label files")

    # val and test must have positives
    for s in all_stats:
        if s.split in ("val", "test") and s.total_images > 0 and s.positive_images == 0:
            errors.append(f"[{s.split}] has 0 positive images — metrics will be invalid")

    if errors:
        for e in errors:
            logger.error(e)
        raise RuntimeError(
            "Dataset validation failed:\n  " + "\n  ".join(errors)
        )

    logger.success("Dataset validation passed — all splits are clean")
    return all_stats