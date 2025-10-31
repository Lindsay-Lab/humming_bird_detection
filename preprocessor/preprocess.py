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
    # In humming_bird_detection/config.py set:
    # PROJ_ROOT = Path(__file__).resolve().parents[1]
    # RAW_DATA_DIR = PROJ_ROOT / "Label-Birdfeeder-Camera-Observations-3"
    # PROCESSED_DATA_DIR = PROJ_ROOT / "data" / "processed"
    # LOGS_DIR = PROJ_ROOT / "logs"
    from humming_bird_detection.config import (
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        LOGS_DIR,
    )
except Exception:
    LOGS_DIR = Path("logs")
    PROCESSED_DATA_DIR = Path("data/processed")
    RAW_DATA_DIR = Path("Label-Birdfeeder-Camera-Observations-3")  # repo_root path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------- Logging ----------
def setup_logging() -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        enqueue=False,
    )
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
    for split in ("train", "val", "test"):
        img_dir = final_root / split / "images"
        lbl_dir = final_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            logger.info(f"[{split}] split missing → skipped checks")
            continue

        img_stems = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
        img_missing_lbl = len(img_stems - lbl_stems)
        lbl_missing_img = len(lbl_stems - img_stems)

        bad_class = bad_range = bad_format = 0
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
            f"[{split}] Sanity | img_without_lbl:{img_missing_lbl} "
            f"| lbl_without_img:{lbl_missing_img} | bad_class:{bad_class} "
            f"| bad_range:{bad_range} | bad_format:{bad_format}"
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

    # Split ratios (industry common: 80/10/10)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


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

    # ---------- Collect all raw pairs ----------
    def _collect_all_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Accepts either Roboflow-style split folders (train/valid/val/test) or a single
        images/labels folder at RAW_DATA_DIR. Returns a flat list of (img,lbl) pairs.
        """
        pairs: List[Tuple[Path, Path]] = []

        # Case A: split folders exist
        split_candidates = ["train_hb", "valid_hb", "val_hb", "test_hb", "train", "valid", "val", "test"]
        found_split = False
        for name in split_candidates:
            p = self.raw_root / name
            if p.exists():
                found_split = True
                img_dir, lbl_dir = p / "images", p / "labels"
                if img_dir.exists() and lbl_dir.exists():
                    pairs.extend(list(self.iter_pairs(img_dir, lbl_dir)))

        # Case B: single folder (images/labels) at root
        if not found_split:
            img_dir, lbl_dir = self.raw_root / "images", self.raw_root / "labels"
            if not (img_dir.exists() and lbl_dir.exists()):
                raise FileNotFoundError(
                    f"Could not find split folders or images/labels under {self.raw_root}."
                )
            pairs.extend(list(self.iter_pairs(img_dir, lbl_dir)))

        if not pairs:
            raise RuntimeError("No (image,label) pairs found in RAW_DATA_DIR.")
        return pairs

    # ---------- Main ----------
    def run(self) -> Path:
        if not self.raw_root.exists():
            raise FileNotFoundError(
                f"RAW_DATA_DIR does not exist: {self.raw_root}\n"
                "Set RAW_DATA_DIR to your repo_root/Label-Birdfeeder-Camera-Observations-3 folder."
            )

        final_root = self.out_root / "hb_ready"
        interim = self.out_root / "interim"
        clean_dir(interim)
        clean_dir(final_root)

        logger.info("Phase 1/3: filter to hummingbird (2→0), sanitize labels, EXIF/RGB → interim/")

        # Build interim by standardizing all files and writing filtered labels
        all_raw_pairs = self._collect_all_pairs()
        # Write everything into one interim bucket
        out_img = interim / "images"
        out_lbl = interim / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        kept = 0
        for img, lbl in tqdm(all_raw_pairs, desc="Standardizing", unit="file", dynamic_ncols=True):
            # remap+filter
            lines = self.filter_and_remap_lines(lbl)
            lines = self.sanitize_lines(lines)

            # EXIF+RGB+JPG
            try:
                with Image.open(img) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    dst_img = out_img / f"{img.stem}.jpg"
                    im.save(dst_img, format="JPEG", quality=95)
            except Exception as e:
                logger.warning(f"Skipping bad image {img}: {e}")
                continue

            # write label (empty allowed)
            dst_lbl = out_lbl / f"{img.stem}.txt"
            if lines:
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                open(dst_lbl, "w").close()
            kept += 1

        logger.info(f"Interim standardized files: {kept}")

        logger.info("Phase 2/3: split → train/val/test (80/10/10) ...")

        # Build pairs from interim (only those with image present; label may be empty)
        interim_pairs: List[Tuple[Path, Path]] = []
        for lbl in sorted((out_lbl).glob("*.txt")):
            img = out_img / (lbl.stem + ".jpg")
            if img.exists():
                interim_pairs.append((img, lbl))

        random.shuffle(interim_pairs)
        n = len(interim_pairs)
        n_train = int(self.cfg.train_ratio * n)
        n_val   = int(self.cfg.val_ratio   * n)
        n_test  = n - n_train - n_val

        train_pairs = interim_pairs[:n_train]
        val_pairs   = interim_pairs[n_train:n_train+n_val]
        test_pairs  = interim_pairs[n_train+n_val:]

        def materialize(pairs: List[Tuple[Path, Path]], split: str) -> int:
            out_i = final_root / split / "images"
            out_l = final_root / split / "labels"
            out_i.mkdir(parents=True, exist_ok=True)
            out_l.mkdir(parents=True, exist_ok=True)
            for img, lbl in tqdm(pairs, desc=f"Copying {split}", unit="file", dynamic_ncols=True):
                shutil.copy2(img, out_i / img.name)
                shutil.copy2(lbl, out_l / lbl.name)
            return len(pairs)

        n_tr = materialize(train_pairs, "train")
        n_va = materialize(val_pairs, "val")
        n_te = materialize(test_pairs, "test")
        logger.success(f"Final split → train:{n_tr} | val:{n_va} | test:{n_te}")

        logger.info("Phase 3/3: write data.yaml + sanity checks")

        data_yaml = final_root / "data.yaml"
        data_yaml.write_text(
            "path: {}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: hummingbird\n".format(
                final_root.resolve()
            )
        )
        logger.success(f"Wrote {data_yaml}")

        sanity_checks(final_root)
        return final_root


# ---------- CLI ----------
def main():
    setup_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_root = RAW_DATA_DIR
    out_root = PROCESSED_DATA_DIR

    cfg = PrepConfig(
        keep_classes_map={2: 0},  # keep Roboflow class 2 as class 0
        min_pixels=2,
        imgsz=640,
        final_img_ext=".jpg",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    final_path = YoloPreprocessor(raw_root, out_root, cfg).run()
    logger.info(f"Preprocessing complete.\nFinal dataset: {final_path}\n→ Use with Ultralytics YOLO.")
    

if __name__ == "__main__":
    main()
