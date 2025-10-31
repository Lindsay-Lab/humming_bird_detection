# preprocess.py
from __future__ import annotations
import os
import shutil
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

from PIL import Image, ImageOps
from loguru import logger
from tqdm import tqdm

# ---------- Paths / Config ----------
try:
    # Optional project config (used for PROCESSED_DATA_DIR, RAW_DATA_DIR, LOGS_DIR)
    from humming_bird_detection.config import (
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        LOGS_DIR,
    )
except Exception:
    LOGS_DIR = Path("logs")
    PROCESSED_DATA_DIR = Path("data/processed")
    RAW_DATA_DIR = Path("data/raw")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Default Roboflow folder name
DEFAULT_RF_DIRNAME = "Label-Birdfeeder-Camera-Observations-3"

# Deterministic order (not a set!)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------- Logging ----------
def setup_logging() -> None:
    logger.remove()
    # Console: flush immediately
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        enqueue=False,
    )
    # File log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOGS_DIR / "preprocess.log", level="INFO", enqueue=True)


# ---------- Helpers ----------
def clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ---------- Sanity checks ----------
def sanity_checks(final_root: Path) -> None:
    """
    - Every image has a label file (possibly empty)
    - Every label file references only class 0
    - YOLO coords are normalized in [0,1]
    """
    img_missing_lbl, lbl_missing_img = 0, 0
    bad_class, bad_range, bad_format = 0, 0, 0

    for split in ("train", "val"):
        img_dir = final_root / split / "images"
        lbl_dir = final_root / split / "labels"
        img_stems = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
        img_missing_lbl += len(img_stems - lbl_stems)
        lbl_missing_img += len(lbl_stems - img_stems)

        for txt in lbl_dir.glob("*.txt"):
            with open(txt, "r") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if not parts:
                        continue  # empty line is fine
                    if len(parts) != 5:
                        bad_format += 1
                        continue
                    c, xc, yc, w, h = parts
                    try:
                        c = int(c); xc = float(xc); yc = float(yc); w = float(w); h = float(h)
                    except Exception:
                        bad_format += 1
                        continue
                    if c != 0:
                        bad_class += 1
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        bad_range += 1

    logger.info(
        f"Sanity | img_without_lbl: {img_missing_lbl}, lbl_without_img: {lbl_missing_img}, "
        f"bad_class: {bad_class}, bad_range: {bad_range}, bad_format: {bad_format}"
    )


# ---------- Config + Preprocessor ----------
@dataclass
class PrepConfig:
    # Keep/remap classes (e.g., {2: 0} to keep only hummingbird and map to 0)
    keep_classes_map: Dict[int, int]

    # Tiny box removal relative to training imgsz
    min_pixels: int = 2
    imgsz: int = 640

    # File/format policy
    final_img_ext: str = ".jpg"

    # Split strategy
    resplit: bool = True          # True: rebuild stratified split; False: reuse RF splits
    split_train_ratio: float = 0.8

    # Optional bottom-crop for timestamp removal
    crop_bottom_pixels: int | None = None   # e.g., 40
    crop_bottom_ratio: float | None = None  # e.g., 0.06 (6% of height)


class YoloPreprocessor:
    def __init__(self, raw_root: Path, out_root: Path, cfg: PrepConfig) -> None:
        self.raw_root = raw_root
        self.out_root = out_root
        self.cfg = cfg

    # ---------- Iterators ----------
    @staticmethod
    def iter_pairs(img_dir: Path, lbl_dir: Path) -> Iterable[Tuple[Path, Path]]:
        for lbl in lbl_dir.glob("*.txt"):
            img = find_image_for_stem(img_dir, lbl.stem)
            if img and lbl.exists():
                yield img, lbl

    # ---------- Label utils ----------
    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def filter_and_remap_lines(self, lbl_path: Path) -> List[str]:
        """
        Read YOLO label file, keep only classes in keep_classes_map, remap to new ids.
        Returns a list of normalized YOLO lines (possibly empty).
        """
        kept: List[str] = []
        try:
            with open(lbl_path, "r") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) != 5:
                        continue
                    c = int(parts[0])
                    if c in self.cfg.keep_classes_map:
                        new_c = self.cfg.keep_classes_map[c]
                        xc, yc, w, h = map(float, parts[1:])
                        kept.append(f"{new_c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        except FileNotFoundError:
            pass
        return kept

    def sanitize_lines(self, lines: List[str]) -> List[str]:
        """
        - clamp normalized coords to [0,1]
        - drop tiny boxes (min_pixels/imgsz)
        """
        min_wh = self.cfg.min_pixels / float(self.cfg.imgsz)
        out: List[str] = []
        for ln in lines:
            c, xc, yc, w, h = ln.split()
            c = int(c)
            xc = self._clamp01(float(xc))
            yc = self._clamp01(float(yc))
            w = max(1e-6, min(1.0, float(w)))
            h = max(1e-6, min(1.0, float(h)))
            if w < min_wh or h < min_wh:
                continue
            out.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        return out

    def _compute_crop_px(self, h: int) -> int:
        if self.cfg.crop_bottom_pixels is not None:
            return max(0, min(int(self.cfg.crop_bottom_pixels), h - 1))
        if self.cfg.crop_bottom_ratio is not None:
            px = int(round(h * float(self.cfg.crop_bottom_ratio)))
            return max(0, min(px, h - 1))
        return 0

    def adjust_labels_for_bottom_crop(
        self, lines: List[str], orig_w: int, orig_h: int, crop_px: int
    ) -> List[str]:
        """
        Given YOLO lines normalized to (orig_w, orig_h), crop 'crop_px' from bottom.
        - Boxes fully inside the cropped band are dropped.
        - Boxes crossing the crop line are clipped to y2' = orig_h - crop_px.
        Returns YOLO-normalized lines for the NEW height (orig_h - crop_px).
        """
        if crop_px <= 0 or not lines:
            return lines

        new_h = orig_h - crop_px
        if new_h <= 1:
            return []  # degenerate image

        adj: List[str] = []
        y_cut = orig_h - crop_px  # absolute pixel y where we cut the bottom off

        for ln in lines:
            c, xc, yc, w, h = ln.split()
            c = int(c)
            xc = float(xc); yc = float(yc); w = float(w); h = float(h)

            # Convert to absolute XYXY on original dims
            x1 = (xc - w / 2.0) * orig_w
            y1 = (yc - h / 2.0) * orig_h
            x2 = (xc + w / 2.0) * orig_w
            y2 = (yc + h / 2.0) * orig_h

            # If the box lies entirely in the cropped-away band, drop it
            if y1 >= y_cut:
                continue

            # Clip to new bottom
            y2 = min(y2, y_cut)

            # Recompute normalized YOLO (note: height is now new_h)
            bw_abs = max(0.0, x2 - x1)
            bh_abs = max(0.0, y2 - y1)
            if bw_abs <= 0 or bh_abs <= 0:
                continue

            xc_abs = x1 + bw_abs / 2.0
            yc_abs = y1 + bh_abs / 2.0

            xc_new = xc_abs / orig_w             # width unchanged
            yc_new = yc_abs / new_h              # height changed
            w_new  = bw_abs / orig_w
            h_new  = bh_abs / new_h

            # Clamp and keep
            xc_new = self._clamp01(xc_new)
            yc_new = self._clamp01(yc_new)
            w_new  = min(1.0, max(1e-6, w_new))
            h_new  = min(1.0, max(1e-6, h_new))
            adj.append(f"{c} {xc_new:.6f} {yc_new:.6f} {w_new:.6f} {h_new:.6f}")

        return adj

    # ---------- Main ----------
    def run(self) -> Path:
        # 0) Locate raw dataset folder robustly (prefer RAW_DATA_DIR, fallback to repo_root)
        candidates: List[Path] = []
        rf_by_cfg = RAW_DATA_DIR / DEFAULT_RF_DIRNAME
        if rf_by_cfg.exists():
            candidates.append(rf_by_cfg)
        rf_by_repo = self.raw_root / DEFAULT_RF_DIRNAME
        if rf_by_repo.exists():
            candidates.append(rf_by_repo)

        if not candidates:
            raise FileNotFoundError(
                f"Roboflow dataset not found. Tried:\n"
                f" - {rf_by_cfg}\n - {rf_by_repo}\n"
                f"Ensure {DEFAULT_RF_DIRNAME} exists under one of these."
            )
        rf_dir = candidates[0]
        logger.info(f"Using dataset at: {rf_dir}")

        # Detect split dirs
        split_dirs: List[Path] = []
        for name in ("train_hb", "valid_hb", "test_hb", "train", "valid", "val", "test"):
            p = rf_dir / name
            if p.exists():
                split_dirs.append(p)
        if not split_dirs:
            raise FileNotFoundError(
                f"No split dirs found under {rf_dir}. Expected e.g. train/, valid/, test/."
            )

        interim = self.out_root / "interim"
        final_root = self.out_root / "hb_ready"
        clean_dir(interim)
        clean_dir(final_root)

        logger.info("Phase 1/3: filtering + EXIF/RGB + optional bottom crop → interim/")

        # 1) Write filtered + normalized interim (no mutation of raw files)
        for split_dir in split_dirs:
            img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            split_tag = split_dir.name
            out_img = interim / split_tag / "images"
            out_lbl = interim / split_tag / "labels"
            safe_mkdir(out_img); safe_mkdir(out_lbl)

            imgs = [p for p in sorted(img_dir.glob("*")) if p.suffix.lower() in IMG_EXTS]
            total = len(imgs)
            nonempty, empty = 0, 0

            logger.info(f"[{split_tag}] starting: {total} images")
            for img in tqdm(imgs, desc=f"Processing {split_tag}", unit="img", dynamic_ncols=True):
                lbl = lbl_dir / f"{img.stem}.txt"

                # Load & remap labels (normalized)
                lines = self.filter_and_remap_lines(lbl)
                # We'll sanitize after possible crop

                # Open image and standardize
                try:
                    with Image.open(img) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        orig_w, orig_h = im.size

                        # Compute crop height and crop pixels (bottom timestamp removal)
                        crop_px = self._compute_crop_px(orig_h)
                        if crop_px > 0:
                            # Adjust labels for crop BEFORE saving
                            lines = self.adjust_labels_for_bottom_crop(lines, orig_w, orig_h, crop_px)
                            # Now apply crop to the image
                            new_h = orig_h - crop_px
                            if new_h <= 1:
                                # Degenerate; skip this image entirely
                                continue
                            im = im.crop((0, 0, orig_w, new_h))

                        # Final sanitize post-crop
                        lines = self.sanitize_lines(lines)

                        # Write image (prefix to avoid collisions across splits)
                        dst_img = out_img / f"{split_tag}__{img.stem}.jpg"
                        im.save(dst_img, format="JPEG", quality=95)
                except Exception as e:
                    logger.warning(f"Skipping bad image {img}: {e}")
                    continue

                # Write label (empty allowed as negative)
                dst_lbl = out_lbl / f"{split_tag}__{img.stem}.txt"
                if lines:
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    nonempty += 1
                else:
                    open(dst_lbl, "w").close()
                    empty += 1

            logger.info(f"[{split_tag}] done: {total} images | nonempty_labels={nonempty}, empty_labels={empty}")

        logger.info("Phase 2/3: build final split → hb_ready/ ...")

        def materialize(pairs: List[Tuple[Path, Path]], split: str) -> int:
            out_img = final_root / split / "images"
            out_lbl = final_root / split / "labels"
            safe_mkdir(out_img); safe_mkdir(out_lbl)
            from shutil import copy2
            for img, lbl in tqdm(pairs, desc=f"Copying {split}", unit="file", dynamic_ncols=True):
                copy2(img, out_img / img.name)
                copy2(lbl, out_lbl / lbl.name)
            return len(pairs)

        if self.cfg.resplit:
            # Build pos/neg list from interim and resplit deterministically
            all_items: List[Tuple[Path, Path, bool]] = []
            for sub_name in sorted(os.listdir(interim)):  # sorted for determinism
                sub = interim / sub_name
                img_dir, lbl_dir = sub / "images", sub / "labels"
                if not (img_dir.exists() and lbl_dir.exists()):
                    continue
                for lbl in sorted(lbl_dir.glob("*.txt")):
                    img = img_dir / (lbl.stem + ".jpg")
                    if not img.exists():
                        continue
                    has_box = False
                    with open(lbl, "r") as f:
                        for ln in f:
                            if ln.strip():
                                has_box = True
                                break
                    all_items.append((img, lbl, has_box))

            pos = [x for x in all_items if x[2]]
            neg = [x for x in all_items if not x[2]]
            random.shuffle(pos); random.shuffle(neg)
            logger.info(f"Interim built: total={len(all_items)} (pos={len(pos)}, neg={len(neg)})")

            def split_group(group, ratio):
                n = len(group); n_tr = int(ratio * n)
                return group[:n_tr], group[n_tr:]

            pos_tr, pos_va = split_group(pos, self.cfg.split_train_ratio)
            neg_tr, neg_va = split_group(neg, self.cfg.split_train_ratio)

            train_pairs = [(i, l) for (i, l, _) in (pos_tr + neg_tr)]
            val_pairs   = [(i, l) for (i, l, _) in (pos_va + neg_va)]

            n_tr = materialize(train_pairs, "train")
            n_va = materialize(val_pairs, "val")

        else:
            # Reuse filtered interim splits as-is
            name_map = {
                "train_hb": "train", "train": "train",
                "valid_hb": "val",   "valid": "val", "val": "val",
                "test_hb":  "val",   "test":  "val",  # merge test into val if present
            }
            totals = {"train": 0, "val": 0}
            for sub_name in sorted(os.listdir(interim)):
                final_name = name_map.get(sub_name)
                if not final_name:
                    continue
                sub = interim / sub_name
                img_dir, lbl_dir = sub / "images", sub / "labels"
                pairs: List[Tuple[Path, Path]] = []
                for lbl in sorted(lbl_dir.glob("*.txt")):
                    img = img_dir / (lbl.stem + ".jpg")
                    if img.exists():
                        pairs.append((img, lbl))
                totals[final_name] += materialize(pairs, final_name)
            n_tr, n_va = totals["train"], totals["val"]

        logger.success(f"Final split → train:{n_tr} | val:{n_va}")

        logger.info("Phase 3/3: writing data.yaml and sanity checks ...")

        # 3) data.yaml
        data_yaml = final_root / "data.yaml"
        data_yaml.write_text(
            "path: {}\ntrain: train/images\nval: val/images\nnames:\n  0: hummingbird\n".format(
                final_root.resolve()
            )
        )
        logger.success(f"Wrote {data_yaml}")

        # 4) sanity checks
        sanity_checks(final_root)

        return final_root


# ---------- CLI ----------
def main():
    setup_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer RAW_DATA_DIR as root; fallback logic inside .run() will locate the dataset folder
    raw_root = RAW_DATA_DIR if RAW_DATA_DIR.exists() else Path(__file__).resolve().parents[1]
    out_root = PROCESSED_DATA_DIR

    cfg = PrepConfig(
        keep_classes_map={2: 0},   # keep Roboflow class 2 as class 0
        min_pixels=2,
        imgsz=640,
        final_img_ext=".jpg",
        resplit=True,              # set False to reuse RF splits after filtering
        split_train_ratio=0.8,
        # >>> set either pixels OR ratio for timestamp band (pixels wins if both set) <<<
        crop_bottom_pixels=40,     # e.g., crop fixed 40 px at bottom
        crop_bottom_ratio=None,    # or e.g., 0.06 for 6% of height
    )

    final_path = YoloPreprocessor(raw_root, out_root, cfg).run()
    logger.info(f"Preprocessing complete.\nFinal dataset: {final_path}\n-> Use this with Ultralytics YOLO.")


if __name__ == "__main__":
    main()
