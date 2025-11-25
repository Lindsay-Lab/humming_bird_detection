from __future__ import annotations
import os
import shutil
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from PIL import Image, ImageOps
from loguru import logger
from tqdm import tqdm

try:
    from humming_bird_detection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, LOGS_DIR
except Exception:
    LOGS_DIR = Path("logs")
    PROCESSED_DATA_DIR = Path("data/processed")
    RAW_DATA_DIR = Path("Label-Birdfeeder-Camera-Observations-3")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ROBoflow_SPLIT_CANDIDATES = ("train_hb", "valid_hb", "val_hb", "test_hb", "train", "valid", "val", "test")

def setup_logging() -> None:
    logger.remove()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(lambda msg: print(msg, end=""),
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
               level="INFO")
    logger.add(LOGS_DIR / "preprocess_yolo.log", level="INFO")

def find_image_for_stem(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_yolo_file(lbl: Path) -> List[List[float]]:
    """Return list of [cls, xc, yc, w, h] for file; empty list if missing/empty."""
    if not lbl.exists():
        return []
    out = []
    with open(lbl, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                c = int(parts[0]); xc, yc, w, h = map(float, parts[1:])
                out.append([c, xc, yc, w, h])
            except Exception:
                continue
    return out

def try_load_data_yaml_names(root: Path) -> Dict[int, str]:
    """Try to load data.yaml under root to map original class ids to names."""
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
    test_ratio: float = 0.2

class YoloPreprocessor:
    def __init__(self, raw_root: Path, out_root: Path, cfg: PrepConfig):
        self.raw_root = raw_root
        self.out_root = out_root
        self.cfg = cfg

    @staticmethod
    def iter_pairs(img_dir: Path, lbl_dir: Path) -> Iterable[Tuple[Path, Path]]:
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                img = find_image_for_stem(img_dir, lbl.stem)
                if img:
                    yield img, lbl
        if img_dir.exists():
            for img in img_dir.glob("*"):
                if img.is_file() and img.suffix.lower() in IMG_EXTS:
                    lbl = lbl_dir / (img.stem + ".txt")
                    if not lbl.exists():
                        yield img, lbl 

    def resolve_hummingbird_ids(self) -> set[int]:
        if self.cfg.hummingbird_ids:
            return set(self.cfg.hummingbird_ids)
        id2name = try_load_data_yaml_names(self.raw_root)
        hb = set()
        for i, nm in id2name.items():
            if nm and any(hint in nm.lower() for hint in self.cfg.hummingbird_name_hints):
                hb.add(i)
        if hb:
            logger.info(f"Resolved hummingbird IDs from data.yaml: {sorted(hb)}")
            return hb
        fallback = {2}
        logger.warning(f"Could not auto-resolve hummingbird IDs; defaulting to {fallback}")
        return fallback

    def _standardize_and_remap(self, img: Path, lbl: Path, out_img_dir: Path, out_lbl_dir: Path, hb_ids: set[int]) -> bool:
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
        found = False
        for name in ROBoflow_SPLIT_CANDIDATES:
            p = self.raw_root / name
            if p.exists() and (p / "images").exists() and (p / "labels").exists():
                found = True
                pairs.extend(list(self.iter_pairs(p / "images", p / "labels")))
        if not found:
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
        safe_mkdir(interim_img); safe_mkdir(interim_lbl)

        kept = 0
        for img, lbl in tqdm(pairs, desc="Standardizing", unit="file", dynamic_ncols=True):
            success = self._standardize_and_remap(img, lbl, interim_img, interim_lbl, hb_ids)
            if success:
                kept += 1
        logger.info(f"Standardized {kept} files into interim/")

        split_present = any((self.raw_root / name).exists() for name in ROBoflow_SPLIT_CANDIDATES)
        manifest = list((interim_lbl).glob("*.txt"))
        manifest = sorted(manifest)
        pairs_final = [(interim_img / (p.stem + self.cfg.final_img_ext), p) for p in manifest if (interim_img / (p.stem + self.cfg.final_img_ext)).exists()]

        if split_present:
            try:
                split_map = {"train": [], "test": []}
                raw_stem_to_split = {}
                for name in ROBoflow_SPLIT_CANDIDATES:
                    p = self.raw_root / name
                    if not p.exists():
                        continue
                    for img in (p / "images").glob("*"):
                        if img.suffix.lower() in IMG_EXTS:
                            raw_stem_to_split[img.stem] = name
                for img_path, lbl_path in pairs_final:
                    st = img_path.stem
                    raw_split = raw_stem_to_split.get(st)
                    if raw_split and raw_split.lower().startswith("train"):
                        split_map["train"].append((img_path, lbl_path))
                    else:
                        split_map["test"].append((img_path, lbl_path))
                if not split_map["train"]:
                    raise RuntimeError("No train mapping inferred; fallback to random split")
                train_pairs = split_map["train"]
                test_pairs = split_map["test"]
            except Exception:
                logger.warning("Could not mirror raw split boundaries; falling back to deterministic random split.")
                random.shuffle(pairs_final)
                n = len(pairs_final)
                n_train = int(self.cfg.train_ratio * n)
                train_pairs = pairs_final[:n_train]
                test_pairs = pairs_final[n_train:]
        else:
            random.shuffle(pairs_final)
            n = len(pairs_final)
            n_train = int(self.cfg.train_ratio * n)
            train_pairs = pairs_final[:n_train]
            test_pairs = pairs_final[n_train:]

        def materialize(pairs: List[Tuple[Path, Path]], split: str) -> int:
            out_img = final_root / split / "images"
            out_lbl = final_root / split / "labels"
            safe_mkdir(out_img); safe_mkdir(out_lbl)
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

        n_tr = materialize(train_pairs, "train")
        n_te = materialize(test_pairs, "test")
        logger.success(f"Final split → train:{n_tr} | test:{n_te}")

        data_yaml = final_root / "data.yaml"
        data_yaml.write_text(
            f"path: {final_root.resolve()}\n"
            "train: train/images\n"
            "test: test/images\n"
            "names:\n  0: hummingbird\n  1: no_hummingbird\n"
        )
        logger.success(f"Wrote {data_yaml}")

        self.sanity_checks(final_root)

        return final_root

    def sanity_checks(self, final_root: Path) -> None:
        for split in ("train", "test"):
            img_dir = final_root / split / "images"
            lbl_dir = final_root / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                logger.info(f"[{split}] missing; skipped sanity check")
                continue
            img_stems = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
            lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
            logger.info(f"[{split}] images={len(img_stems)} labels={len(lbl_stems)}")

            c0 = c1 = empty = bad = 0
            for lbl in lbl_dir.glob("*.txt"):
                lines = read_yolo_file(lbl)
                if not lines:
                    empty += 1
                else:
                    for l in lines:
                        if int(l[0]) == 0:
                            c0 += 1
                        elif int(l[0]) == 1:
                            c1 += 1
                        else:
                            bad += 1
            logger.info(f"[{split}] boxes -> hummingbird(0):{c0} other(1):{c1} empty_files:{empty} bad:{bad}")

def main():
    setup_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cfg = PrepConfig()

    preprocessor = YoloPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR, cfg)
    final = preprocessor.run()
    logger.success(f"Preprocessing complete. Final dataset at: {final}")

if __name__ == "__main__":
    main()
