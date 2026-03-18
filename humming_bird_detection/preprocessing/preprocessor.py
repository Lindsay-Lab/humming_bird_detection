"""Preprocessing orchestrator.

Pipeline (9 steps):
 1. Collect all raw image/label pairs from Roboflow splits.
 2. Flatten into a single interim directory (deduplicate).
 3. Crop timestamp bars from camera images; save strips separately.
 4. Sanitize labels — drop invalid rows (w<=0, h<=0, out-of-range).
 5. Standardize images — RGB, long-side downscale, JPEG (parallel).
    (optional) Visualize after standardization for quality check.
 6. Background separation — pretrained YOLO26n-COCO detection inference.
 7. Remap classes — hummingbird -> 0, other_bird -> 1.
    (remap AFTER bg separation; bg sep uses COCO model, not our labels)
 8. Split positive (non-background) images into train/val/test (80/10/10).
 9. Write data.yaml, generate metadata CSV, visualize final samples.

No offline augmentation — will use YOLO's built-in augmentations during training
(mosaic, HSV jitter, flipping, scaling, translation).

Output directory structure:
    processed/hb_detection/hb_ready/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
        data.yaml
    processed/hb_detection/backgrounds/images/  labels/
    processed/hb_detection/timestamp_strips/
    processed/hb_detection/visualizations/
    processed/hb_detection/image_metadata.csv
"""
from __future__ import annotations

import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from humming_bird_detection.configs.config import PrepConfig
from humming_bird_detection.configs.paths import LOGS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from humming_bird_detection.preprocessing.build_yolo import remap_labels, resolve_hummingbird_ids
from humming_bird_detection.preprocessing.crop_timestamp import crop_timestamp_dataset
from humming_bird_detection.preprocessing.separate_backgrounds import separate_backgrounds
from humming_bird_detection.preprocessing.standardize import standardize_dataset
from humming_bird_detection.preprocessing.validate import read_yolo_lines, sanitize_labels
from humming_bird_detection.preprocessing.visualize import visualize_dir, visualize_preprocessed
from humming_bird_detection.utils.logging import setup_logging

IMG_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _find_image_for_stem(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def _iter_pairs(img_dir: Path, lbl_dir: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (image, label) pairs from matching images/labels directories."""
    yielded: set = set()
    if lbl_dir.exists():
        for lbl in lbl_dir.glob("*.txt"):
            img = _find_image_for_stem(img_dir, lbl.stem)
            if img:
                yielded.add(img.stem)
                yield img, lbl
    if img_dir.exists():
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS and img.stem not in yielded:
                lbl = lbl_dir / f"{img.stem}.txt"
                yield img, lbl

def _collect_all_pairs(raw_root: Path) -> List[Tuple[Path, Path]]:
    """Merge all Roboflow splits into a single flat list of (img, lbl) pairs."""
    pairs: List[Tuple[Path, Path]] = []
    for split_name in ("train", "valid", "test"):
        img_dir = raw_root / split_name / "images"
        lbl_dir = raw_root / split_name / "labels"
        if img_dir.exists():
            split_pairs = list(_iter_pairs(img_dir, lbl_dir))
            logger.info(f"  [{split_name}] found {len(split_pairs)} pairs")
            pairs.extend(split_pairs)
    # Fallback: flat layout
    if not pairs:
        img_dir = raw_root / "images"
        lbl_dir = raw_root / "labels"
        if img_dir.exists():
            pairs.extend(list(_iter_pairs(img_dir, lbl_dir)))
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {raw_root}")
    logger.info(f"Total collected: {len(pairs)} image/label pairs")
    return pairs

def _flatten_to_dir(
    pairs: List[Tuple[Path, Path]],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> int:
    """Copy pairs into flat images/ + labels/ directories. Deduplicates by stem."""
    _safe_mkdir(dst_img_dir)
    _safe_mkdir(dst_lbl_dir)
    seen: set = set()
    copied = 0
    for img, lbl in tqdm(pairs, desc="Flattening", unit="file", dynamic_ncols=True):
        if img.stem in seen:
            continue
        seen.add(img.stem)
        try:
            shutil.copy2(img, dst_img_dir / img.name)
        except Exception as exc:
            logger.warning(f"Copy failed {img.name}: {exc}")
            continue
        dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.write_text("")
        copied += 1
    logger.info(f"Flattened {copied} unique pairs")
    return copied

def _split_positive_images(
    img_dir: Path,
    lbl_dir: Path,
    final_root: Path,
    cfg: PrepConfig,
) -> Dict[str, int]:
    """Split positive images into train/val/test (80/10/10). Copy to final dirs.
    Returns dict of split counts.
    """
    random.seed(cfg.seed)
    # Collect all image/label pairs
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    pairs = [(img, lbl_dir / f"{img.stem}.txt") for img in images]

    total = len(pairs)
    n_val = max(1, int(round(cfg.val_ratio * total)))
    n_test = max(1, int(round(cfg.test_ratio * total)))
    n_train = total - n_val - n_test
    if n_train <= 0:
        raise RuntimeError(f"Not enough images to split: total={total}, val={n_val}, test={n_test}")

    random.shuffle(pairs)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    logger.info(
        f"Split ratios: train={cfg.train_ratio}, val={cfg.val_ratio}, test={cfg.test_ratio} | "
        f"Counts: train={n_train}, val={n_val}, test={n_test} (total={total})"
    )

    counts = {}
    for split_name, split_pairs in splits.items():
        out_img = final_root / split_name / "images"
        out_lbl = final_root / split_name / "labels"
        _safe_mkdir(out_img)
        _safe_mkdir(out_lbl)
        for img, lbl in split_pairs:
            try:
                os.link(img, out_img / img.name)
            except Exception:
                shutil.copy2(img, out_img / img.name)
            if lbl.exists():
                try:
                    os.link(lbl, out_lbl / lbl.name)
                except Exception:
                    shutil.copy2(lbl, out_lbl / lbl.name)
            else:
                (out_lbl / f"{img.stem}.txt").write_text("")
        counts[split_name] = len(split_pairs)
        logger.info(f"  [{split_name}] {len(split_pairs)} images -> {out_img}")

    return counts

def _write_data_yaml(final_root: Path, splits: List[str], cfg: PrepConfig) -> Path:
    data_yaml = final_root / "data.yaml"
    lines = [f"path: {final_root.resolve()}"]
    for s in splits:
        lines.append(f"{s}: {s}/images")
    lines.append(f"nc: {len(cfg.class_names)}")
    names_block = "names:"
    for i, name in enumerate(cfg.class_names):
        names_block += f"\n  {i}: {name}"
    lines.append(names_block)
    data_yaml.write_text("\n".join(lines) + "\n")
    logger.success(f"Wrote {data_yaml}")
    return data_yaml

def _write_metadata_csv(
    base_dir: Path,
    final_root: Path,
    bg_dir: Path,
    ts_strip_dir: Path,
    bg_counts: Dict,
    split_counts: Dict[str, int],
    cfg: PrepConfig,
) -> Path:
    """Write image_metadata.csv mapping each image to its processing result.

    Columns: filename, split, bg_separation_result, class_counts, timestamp_strip_path
    This lets you cross-reference: image → timestamp → detection result → class.
    """
    csv_path = base_dir / "image_metadata.csv"
    rows = []

    # Final dataset images (positive, split into train/val/test)
    for split_name in ("train", "val", "test"):
        split_img_dir = final_root / split_name / "images"
        split_lbl_dir = final_root / split_name / "labels"
        if not split_img_dir.exists():
            continue
        for img in sorted(split_img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = split_lbl_dir / f"{img.stem}.txt"
            class_counts = {}
            if lbl.exists():
                for ln in lbl.read_text().strip().splitlines():
                    parts = ln.strip().split()
                    if len(parts) >= 1:
                        try:
                            c = int(parts[0])
                            cname = cfg.class_names[c] if c < len(cfg.class_names) else f"cls_{c}"
                            class_counts[cname] = class_counts.get(cname, 0) + 1
                        except (ValueError, IndexError):
                            pass
            ts_strip = ts_strip_dir / img.name
            rows.append({
                "filename": img.name,
                "split": split_name,
                "bg_separation_result": "positive",
                "class_counts": str(class_counts) if class_counts else "empty_label",
                "timestamp_strip_exists": ts_strip.exists(),
                "timestamp_strip_path": str(ts_strip) if ts_strip.exists() else "",
            })

    # Background images
    bg_img_dir = bg_dir / "images"
    if bg_img_dir.exists():
        for img in sorted(bg_img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            ts_strip = ts_strip_dir / img.name
            rows.append({
                "filename": img.name,
                "split": "background",
                "bg_separation_result": "background",
                "class_counts": "no_bird_detected",
                "timestamp_strip_exists": ts_strip.exists(),
                "timestamp_strip_path": str(ts_strip) if ts_strip.exists() else "",
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "split", "bg_separation_result",
            "class_counts", "timestamp_strip_exists", "timestamp_strip_path",
        ])
        writer.writeheader()
        writer.writerows(rows)

    logger.success(f"Wrote metadata CSV: {csv_path} ({len(rows)} rows)")
    return csv_path

def _log_dir_tree(root: Path, depth: int = 3) -> None:
    """Log directory structure up to given depth."""
    def _walk(p: Path, level: int):
        if level > depth:
            return
        indent = "  " * level
        if p.is_dir():
            n_files = sum(1 for f in p.iterdir() if f.is_file())
            n_dirs = sum(1 for f in p.iterdir() if f.is_dir())
            logger.info(f"{indent}{p.name}/ ({n_files} files, {n_dirs} subdirs)")
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    _walk(child, level + 1)
    logger.info(f"Output directory structure: {root}")
    _walk(root, 0)

class YoloPreprocessor:

    def __init__(self, raw_root: Path, out_root: Path, cfg: PrepConfig):
        self.raw_root = raw_root
        self.out_root = out_root
        self.cfg = cfg

    def run(self) -> Path:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Raw data not found: {self.raw_root}")

        base = self.out_root / "hb_detection"
        final_root = base / "hb_ready"
        interim = base / "interim"
        standardized = base / "standardized"
        bg_dir = base / "backgrounds"
        ts_strip_dir = base / "timestamp_strips"
        viz_dir = base / "visualizations"

        # Clean previous runs
        for d in (interim, standardized, final_root, bg_dir, ts_strip_dir, viz_dir):
            if d.exists():
                shutil.rmtree(d)

        # Collect all raw pairs
        logger.info("=" * 60)
        logger.info("STEP 1/9: Collecting all raw image/label pairs")
        pairs = _collect_all_pairs(self.raw_root)

        #Flatten into interim
        logger.info("=" * 60)
        logger.info("STEP 2/9: Flattening into interim directory")
        interim_img = interim / "images"
        interim_lbl = interim / "labels"
        _flatten_to_dir(pairs, interim_img, interim_lbl)

        # Crop timestamp bars
        logger.info("=" * 60)
        if self.cfg.crop_timestamp:
            logger.info(
                f"STEP 3/9: Cropping timestamp bars "
                f"(bottom {self.cfg.timestamp_crop_fraction*100:.2f}% of image)"
            )
            cropped_img = interim / "cropped_images"
            cropped_lbl = interim / "cropped_labels"
            crop_ok, crop_fail = crop_timestamp_dataset(
                src_img_dir=interim_img,
                src_lbl_dir=interim_lbl,
                dst_img_dir=cropped_img,
                dst_lbl_dir=cropped_lbl,
                timestamp_strip_dir=ts_strip_dir,
                cfg=self.cfg,
            )
            logger.info(f"Timestamp crop result: {crop_ok} cropped, {crop_fail} failed")
            logger.info(f"Timestamp strips saved to: {ts_strip_dir}")
            interim_img = cropped_img
            interim_lbl = cropped_lbl
        else:
            logger.info("STEP 3/9: Timestamp cropping SKIPPED (crop_timestamp=False)")

        # Sanitize labels
        logger.info("=" * 60)
        logger.info("STEP 4/9: Sanitizing labels (dropping invalid rows)")
        sanitize_labels(interim_lbl, self.cfg, fix_in_place=True)

        # Standardize images (parallel)
        logger.info("=" * 60)
        logger.info("STEP 5/9: Standardizing images (RGB + long-side downscale + JPEG)")
        std_img = standardized / "images"
        std_lbl = standardized / "labels"
        kept, skipped = standardize_dataset(
            src_img_dir=interim_img,
            src_lbl_dir=interim_lbl,
            dst_img_dir=std_img,
            dst_lbl_dir=std_lbl,
            cfg=self.cfg,
        )
        logger.info(f"Standardization result: {kept} kept, {skipped} skipped")

        # Visualize after standardize (quality check)
        logger.info("Generating post-standardization visualizations...")
        visualize_dir(
            img_dir=std_img, lbl_dir=std_lbl,
            output_dir=viz_dir, tag="after_standardize",
            n_samples=self.cfg.visualize_samples, seed=self.cfg.seed,
        )

        # Background separation
        logger.info("=" * 60)
        logger.info(
            "STEP 6/9: Separating backgrounds (pretrained YOLO26n-COCO detection)\n"
            "  Why COCO? COCO is an object-detection dataset (80 classes incl. 'bird').\n"
            "  ImageNet is classification-only — it can't locate objects in a scene.\n"
            "  We need detection to answer 'is a bird present?' not 'is the whole image a bird?'"
        )
        positive_img = interim / "positive" / "images"
        positive_lbl = interim / "positive" / "labels"
        bg_img = bg_dir / "images"
        bg_lbl = bg_dir / "labels"
        bg_counts = separate_backgrounds(
            img_dir=std_img,
            lbl_dir=std_lbl,
            positive_img_dir=positive_img,
            positive_lbl_dir=positive_lbl,
            background_img_dir=bg_img,
            background_lbl_dir=bg_lbl,
            cfg=self.cfg,
        )
        logger.info(
            f"Background separation: positive={bg_counts['positive']}, "
            f"background={bg_counts['background']}, failed={bg_counts['failed']}"
        )
        logger.info(f"Background images saved to: {bg_dir}")

        # Remap classes (AFTER bg separation)
        logger.info("=" * 60)
        logger.info(
            "STEP 7/9: Remapping classes (hummingbird=0, other_bird=1)\n"
            "  Remap happens AFTER bg separation because bg sep uses a pretrained\n"
            "  COCO model on image pixels — it does not read our label files."
        )
        hb_ids = resolve_hummingbird_ids(self.raw_root, self.cfg)
        remapped_lbl = interim / "positive_remapped" / "labels"
        remap_labels(positive_lbl, remapped_lbl, hb_ids, num_workers=self.cfg.num_workers)

        # Split positive images (80/10/10)
        logger.info("=" * 60)
        logger.info(
            f"STEP 8/9: Splitting positive images into train/val/test "
            f"({self.cfg.train_ratio}/{self.cfg.val_ratio}/{self.cfg.test_ratio})"
        )
        split_counts = _split_positive_images(
            img_dir=positive_img,
            lbl_dir=remapped_lbl,
            final_root=final_root,
            cfg=self.cfg,
        )
        logger.success(
            "Split complete: " + " | ".join(f"{k}={v}" for k, v in split_counts.items())
        )

        # Finalize
        logger.info("=" * 60)
        logger.info("STEP 9/9: Writing data.yaml, metadata CSV, final visualizations")
        logger.info(
            "No offline augmentation applied — use YOLO's built-in augmentations\n"
            "  during training (mosaic, HSV jitter, flip, scale, translation).\n"
            "  This is more effective for detection and avoids dataset bloat."
        )

        active_splits = [s for s in ("train", "val", "test") if split_counts.get(s, 0) > 0]
        _write_data_yaml(final_root, active_splits, self.cfg)

        _write_metadata_csv(
            base_dir=base,
            final_root=final_root,
            bg_dir=bg_dir,
            ts_strip_dir=ts_strip_dir,
            bg_counts=bg_counts,
            split_counts=split_counts,
            cfg=self.cfg,
        )

        visualize_preprocessed(final_root, viz_dir, self.cfg)
        logger.info(f"Sample visualizations saved to: {viz_dir}")

        # Cleanup
        for d in (interim, standardized):
            if d.exists():
                shutil.rmtree(d)
        logger.info("Cleaned up interim/standardized directories")

        # Final summary
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE — SUMMARY")
        logger.info(f"  Raw input:       {self.raw_root}")
        logger.info(f"  Final dataset:   {final_root}")
        logger.info(f"  Backgrounds:     {bg_dir}")
        logger.info(f"  Timestamp strips:{ts_strip_dir}")
        logger.info(f"  Metadata CSV:    {base / 'image_metadata.csv'}")
        logger.info(f"  Visualizations:  {viz_dir}")
        logger.info(f"  Splits: {split_counts}")
        logger.info(f"  Class names: {dict(enumerate(self.cfg.class_names))}")
        logger.info(f"  Batch size:  {self.cfg.batch_size}")
        logger.info("  Augmentation: NONE (use YOLO built-in during training)")
        _log_dir_tree(base, depth=3)

        logger.success(f"Pipeline complete. Final dataset: {final_root}")
        return final_root

def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Preprocessing Pipeline")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Path to raw dataset (default: from paths.py)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Path to output processed data (default: from paths.py)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (saved in data config)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--no-crop-timestamp", action="store_true",
                        help="Skip timestamp bar cropping")
    parser.add_argument("--max-long-side", type=int, default=None,
                        help="Max long side for image downscaling")
    args = parser.parse_args()

    setup_logging(LOGS_DIR, "preprocess", "DEBUG")

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DATA_DIR
    out_dir = Path(args.out_dir) if args.out_dir else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PrepConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.no_crop_timestamp:
        cfg.crop_timestamp = False
    if args.max_long_side is not None:
        cfg.max_long_side = args.max_long_side

    logger.info(f"RAW_DATA_DIR  = {raw_dir}")
    logger.info(f"PROCESSED_DIR = {out_dir}")
    logger.info(f"LOGS_DIR      = {LOGS_DIR}")
    logger.info(f"Config: {cfg}")

    preprocessor = YoloPreprocessor(raw_dir, out_dir, cfg)
    final = preprocessor.run()
    logger.success(f"Preprocessing complete. Final dataset at: {final}")

if __name__ == "__main__":
    main()