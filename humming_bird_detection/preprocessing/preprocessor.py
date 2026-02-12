from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm

from humming_bird_detection.config import LOGS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        LOGS_DIR / "preprocess_yolo.log",
        level="INFO",
        format="<green>{time}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
    )


def find_image_for_stem(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_yolo_file(lbl: Path) -> List[List[float]]:
    if not lbl.exists():
        return []
    out: List[List[float]] = []
    with open(lbl, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                c = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                out.append([c, xc, yc, w, h])
            except Exception:
                continue
    return out


def try_load_data_yaml_names(root: Path) -> Dict[int, str]:
    try:
        import yaml
    except Exception:
        logger.debug("PyYAML not available — skipping data.yaml lookup.")
        return {}

    candidates = [root / "data.yaml", root / "dataset.yaml", root / "data" / "data.yaml"]
    for c in candidates:
        if c.exists():
            try:
                d = yaml.safe_load(c.read_text())
                names = d.get("names")
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
                if isinstance(names, list):
                    return {i: n for i, n in enumerate(names)}
            except Exception as e:
                logger.debug(f"Failed parsing {c}: {e}")
                continue
    return {}


@dataclass
class PrepConfig:
    hummingbird_ids: Optional[set[int]] = None
    hummingbird_name_hints: Tuple[str, ...] = ("humming", "hummingbird")
    final_img_ext: str = ".jpg"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    bg_ratio_val: float = 0.05   # fraction of *val labeled count* to add as backgrounds
    bg_ratio_test: float = 0.05  # fraction of *test labeled count* to add as backgrounds


class YoloPreprocessor:
    def __init__(self, raw_root: Path, out_root: Path, cfg: PrepConfig):
        self.raw_root = raw_root
        self.out_root = out_root
        self.cfg = cfg

    @staticmethod
    def iter_pairs(img_dir: Path, lbl_dir: Path) -> Iterable[Tuple[Path, Path]]:
        yielded = set()
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                img = find_image_for_stem(img_dir, lbl.stem)
                if img:
                    yielded.add(img.stem)
                    yield img, lbl
        if img_dir.exists():
            for img in img_dir.glob("*"):
                if img.suffix.lower() in IMG_EXTS and img.stem not in yielded:
                    lbl = lbl_dir / f"{img.stem}.txt"
                    yield img, lbl

    def resolve_hummingbird_ids(self) -> set[int]:
        # CHANGE 1: remove dangerous fallback; require explicit IDs if yaml can't resolve
        if self.cfg.hummingbird_ids:
            return set(self.cfg.hummingbird_ids)

        id2name = try_load_data_yaml_names(self.raw_root)
        hb: set[int] = set()
        for i, nm in id2name.items():
            if nm and any(hint in nm.lower() for hint in self.cfg.hummingbird_name_hints):
                hb.add(i)

        if hb:
            logger.info(f"Resolved hummingbird IDs from data.yaml: {sorted(hb)}")
            return hb

        raise RuntimeError(
            "Could not auto-resolve hummingbird class IDs from any data.yaml/dataset.yaml. "
            "Provide PrepConfig(hummingbird_ids={...}) explicitly to avoid silent mislabeling."
        )

    def _standardize_and_remap(
        self,
        img: Path,
        lbl: Path,
        out_img_dir: Path,
        out_lbl_dir: Path,
        hb_ids: set[int],
    ) -> bool:
        boxes = read_yolo_file(lbl)
        remapped_lines: List[str] = []
        for b in boxes:
            orig_c, xc, yc, w, h = int(b[0]), b[1], b[2], b[3], b[4]
            new_c = 0 if orig_c in hb_ids else 1
            remapped_lines.append(f"{new_c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        try:
            safe_mkdir(out_img_dir)
            with Image.open(img) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                out_img = out_img_dir / f"{img.stem}{self.cfg.final_img_ext}"
                im.save(out_img, format="JPEG", quality=95)
        except Exception as e:
            logger.warning(f"Skipping image {img}: {e}")
            return False

        safe_mkdir(out_lbl_dir)
        out_lbl = out_lbl_dir / f"{img.stem}.txt"
        if remapped_lines:
            out_lbl.write_text("\n".join(remapped_lines) + "\n")
        else:
            out_lbl.write_text("")

        return True

    def collect_raw_pairs(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        img_dir = self.raw_root / "images"
        lbl_dir = self.raw_root / "labels"
        if img_dir.exists() and lbl_dir.exists():
            pairs.extend(list(self.iter_pairs(img_dir, lbl_dir)))
        if not pairs:
            raise RuntimeError(f"No image/label pairs found under {self.raw_root}")
        return pairs

    def run(self) -> Path:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"RAW_DATA_DIR not found: {self.raw_root}")

        final_root = self.out_root / "hb_detection" / "hb_ready"
        interim = self.out_root / "hb_detection" / "interim"

        if interim.exists():
            shutil.rmtree(interim)
        if final_root.exists():
            shutil.rmtree(final_root)
        safe_mkdir(interim)
        safe_mkdir(final_root)

        hb_ids = self.resolve_hummingbird_ids()
        logger.info(f"Hummingbird original IDs: {sorted(hb_ids)} -> remapped to 0; others -> 1")

        pairs = self.collect_raw_pairs()
        logger.info(f"Found {len(pairs)} image candidates (labels may be empty)")

        interim_img = interim / "images"
        interim_lbl = interim / "labels"
        safe_mkdir(interim_img)
        safe_mkdir(interim_lbl)

        kept = 0
        for img, lbl in tqdm(pairs, desc="Standardizing", unit="file", dynamic_ncols=True):
            success = self._standardize_and_remap(img, lbl, interim_img, interim_lbl, hb_ids)
            if success:
                kept += 1
        logger.info(f"Standardized {kept} files into interim/")

        manifest = sorted(list(interim_lbl.glob("*.txt")))
        pairs_final = [
            (interim_img / (p.stem + self.cfg.final_img_ext), p)
            for p in manifest
            if (interim_img / (p.stem + self.cfg.final_img_ext)).exists()
        ]

        labeled_pairs: List[Tuple[Path, Path]] = []
        unlabeled_pairs: List[Tuple[Path, Path]] = []
        for img_path, lbl_path in pairs_final:
            if read_yolo_file(lbl_path):
                labeled_pairs.append((img_path, lbl_path))
            else:
                unlabeled_pairs.append((img_path, lbl_path))

        logger.info(f"Labeled samples: {len(labeled_pairs)}, Unlabeled samples: {len(unlabeled_pairs)}")

        total_labeled_pool = len(labeled_pairs)
        if total_labeled_pool == 0:
            raise RuntimeError("No labeled samples found after preprocessing; cannot create splits.")

        # CHANGE 2: robust split sizing with guardrail: n_val + n_test <= total_labeled - 1
        n_val = max(1, int(round(self.cfg.val_ratio * total_labeled_pool)))
        n_test = max(1, int(round(self.cfg.test_ratio * total_labeled_pool)))

        if n_val + n_test >= total_labeled_pool:
            overflow = (n_val + n_test) - (total_labeled_pool - 1)
            if overflow > 0:
                # shrink test first, but keep >=1 if possible
                shrink = min(overflow, max(0, n_test - 1))
                n_test -= shrink
                overflow -= shrink
            if overflow > 0:
                n_val = max(1, n_val - overflow)

        n_train = total_labeled_pool - n_val - n_test
        if n_train <= 0:
            raise RuntimeError(
                f"Invalid split sizes computed: train={n_train}, val={n_val}, test={n_test}, "
                f"total_labeled={total_labeled_pool}. Adjust ratios or provide more labeled data."
            )

        expected_val_pos = n_val
        expected_test_pos = n_test
        expected_train_pos = n_train

        logger.info(
            f"Split sizes (labeled pool) - Train: {n_train}, Val: {n_val}, Test: {n_test} "
            f"(from {total_labeled_pool} labeled samples)"
        )

        random.shuffle(labeled_pairs)
        train_pairs = labeled_pairs[:n_train]
        val_pairs = labeled_pairs[n_train : n_train + n_val]
        test_pairs = labeled_pairs[n_train + n_val : n_train + n_val + n_test]

        # Add small controlled negatives to val/test (optional; can be 0)
        random.shuffle(unlabeled_pairs)
        n_val_bg = int(round(self.cfg.bg_ratio_val * expected_val_pos))
        n_test_bg = int(round(self.cfg.bg_ratio_test * expected_test_pos))

        # Clamp to available
        n_val_bg = min(n_val_bg, len(unlabeled_pairs))
        n_test_bg = min(n_test_bg, max(0, len(unlabeled_pairs) - n_val_bg))

        if n_val_bg or n_test_bg:
            logger.info(f"Adding background images - Val: {n_val_bg}, Test: {n_test_bg}")
            val_bg_pairs = unlabeled_pairs[:n_val_bg]
            test_bg_pairs = unlabeled_pairs[n_val_bg : n_val_bg + n_test_bg]
            remaining_bg = unlabeled_pairs[n_val_bg + n_test_bg :]

            val_pairs.extend(val_bg_pairs)
            test_pairs.extend(test_bg_pairs)
            train_pairs.extend(remaining_bg)
        else:
            if unlabeled_pairs:
                logger.info("No background images added to val/test (bg ratios rounded to 0); adding all backgrounds to train.")
                train_pairs.extend(unlabeled_pairs)
            else:
                logger.info("No unlabeled (background) images available; splits will be labeled-only.")

        split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        def materialize(pairs: List[Tuple[Path, Path]], split: str) -> int:
            out_img = final_root / split / "images"
            out_lbl = final_root / split / "labels"
            safe_mkdir(out_img)
            safe_mkdir(out_lbl)
            for img, lbl in tqdm(pairs, desc=f"Copying {split}", unit="file", dynamic_ncols=True):
                dst_img = out_img / img.name
                try:
                    os.link(img, dst_img)
                except Exception:
                    shutil.copy2(img, dst_img)

                dst_lbl = out_lbl / lbl.name
                try:
                    os.link(lbl, dst_lbl)
                except Exception:
                    shutil.copy2(lbl, dst_lbl)
            return len(pairs)

        split_counts = {}
        for split_name, pairs_ in split_map.items():
            split_counts[split_name] = materialize(pairs_, split_name)
        logger.success("Final splits → " + " | ".join(f"{k}:{v}" for k, v in split_counts.items()))

        # CHANGE 3: write data.yaml WITHOUT test if test is empty
        data_yaml = final_root / "data.yaml"
        lines = [f"path: {final_root.resolve()}"]
        lines.append("train: train/images")
        lines.append("val: val/images")
        if len(test_pairs) > 0:
            lines.append("test: test/images")
        else:
            logger.warning("Test split is empty; not writing 'test:' entry to data.yaml.")
        lines.append("names:\n  0: hummingbird\n  1: no_hummingbird")
        data_yaml.write_text("\n".join(lines))
        logger.success(f"Wrote {data_yaml}")

        # CHANGE 4: pass pre-split expectations into sanity_checks
        self.sanity_checks(
            final_root=final_root,
            total_labeled_pool=total_labeled_pool,
            expected_train_pos=expected_train_pos,
            expected_val_pos=expected_val_pos,
            expected_test_pos=expected_test_pos,
            expected_val_bg=n_val_bg,
            expected_test_bg=n_test_bg,
        )

        return final_root

    def sanity_checks(
        self,
        final_root: Path,
        total_labeled_pool: int,
        expected_train_pos: int,
        expected_val_pos: int,
        expected_test_pos: int,
        expected_val_bg: int,
        expected_test_bg: int,
    ) -> None:
        split_stats: Dict[str, Dict[str, int]] = {}

        for split in ("train", "val", "test"):
            img_dir = final_root / split / "images"
            lbl_dir = final_root / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                logger.info(f"[{split}] missing; skipped sanity check")
                continue

            img_stems = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
            lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
            logger.info(f"[{split}] images={len(img_stems)} labels={len(lbl_stems)}")

            c0 = c1 = empty = bad = 0
            labeled_count = 0
            for lbl in lbl_dir.glob("*.txt"):
                lines = read_yolo_file(lbl)
                if not lines:
                    empty += 1
                else:
                    labeled_count += 1
                    for l in lines:
                        if int(l[0]) == 0:
                            c0 += 1
                        elif int(l[0]) == 1:
                            c1 += 1
                        else:
                            bad += 1

            split_stats[split] = {
                "total": len(lbl_stems),
                "labeled": labeled_count,  # non-empty label files (positive images)
                "empty": empty,            # background images
                "c0": c0,
                "c1": c1,
                "bad": bad,
            }
            logger.info(f"[{split}] boxes -> hummingbird(0):{c0} other(1):{c1} empty_files:{empty} bad:{bad}")

        # CHANGE 5: verify against the ORIGINAL labeled pool expectations (not sum across splits)
        logger.info(f"Split verification (base=labeled pool) - Total labeled pool: {total_labeled_pool}")
        logger.info(
            f"Expected positives - Train: {expected_train_pos}, Val: {expected_val_pos}, Test: {expected_test_pos}"
        )
        logger.info(f"Expected backgrounds - Val: {expected_val_bg}, Test: {expected_test_bg}")

        train_pos = split_stats.get("train", {}).get("labeled", 0)
        val_pos = split_stats.get("val", {}).get("labeled", 0)
        test_pos = split_stats.get("test", {}).get("labeled", 0)

        val_bg = split_stats.get("val", {}).get("empty", 0)
        test_bg = split_stats.get("test", {}).get("empty", 0)

        if train_pos != expected_train_pos:
            logger.warning(f"Train split has {train_pos} positive images, expected {expected_train_pos}")
        else:
            logger.success(f"Train split has correct {train_pos} positive images")

        if val_pos != expected_val_pos:
            logger.warning(f"Val split has {val_pos} positive images, expected {expected_val_pos}")
        else:
            logger.success(f"Val split has correct {val_pos} positive images")

        if expected_test_pos > 0:
            if test_pos != expected_test_pos:
                logger.warning(f"Test split has {test_pos} positive images, expected {expected_test_pos}")
            else:
                logger.success(f"Test split has correct {test_pos} positive images")
        else:
            logger.info("No test positives expected (test split empty).")

        # Background checks (exact match is reasonable since we compute them deterministically)
        if val_bg != expected_val_bg:
            logger.warning(f"Val split has {val_bg} background images, expected {expected_val_bg}")
        else:
            logger.success(f"Val split has correct {val_bg} background images")

        if test_bg != expected_test_bg:
            logger.warning(f"Test split has {test_bg} background images, expected {expected_test_bg}")
        else:
            logger.success(f"Test split has correct {test_bg} background images")

        # Ensure val/test contain at least one labeled sample
        if val_pos == 0:
            logger.error("Val split has no labeled (positive) samples — metrics will be invalid.")
        if expected_test_pos > 0 and test_pos == 0:
            logger.error("Test split has no labeled (positive) samples — metrics will be invalid.")

        # Distribution info (positives only)
        val_pct = (val_pos / total_labeled_pool) * 100 if total_labeled_pool else 0.0
        test_pct = (test_pos / total_labeled_pool) * 100 if total_labeled_pool else 0.0
        logger.info(f"Positive-only distribution - Val: {val_pct:.1f}%, Test: {test_pct:.1f}%")

        logger.info("Final split composition:")
        logger.info(f"  Train: {train_pos} positive + {split_stats.get('train', {}).get('empty', 0)} background")
        logger.info(f"  Val:   {val_pos} positive + {val_bg} background")
        logger.info(f"  Test:  {test_pos} positive + {test_bg} background")


def main():
    setup_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"RAW_DATA_DIR = {RAW_DATA_DIR}")

    cfg = PrepConfig()
    preprocessor = YoloPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR, cfg)
    final = preprocessor.run()
    logger.success(f"Preprocessing complete. Final dataset at: {final}")


if __name__ == "__main__":
    main()