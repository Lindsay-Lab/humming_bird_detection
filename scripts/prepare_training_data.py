"""Prepare the classifier training directory layout.

Supports two input modes:

  Mode A — Roboflow YOLOv8/v11 download (recommended):
    Reads a directory produced by ``version.download("yolov11")``.
    Pools images from all Roboflow splits (train/ + valid/ + test/), ignoring
    Roboflow's own split assignment.  For each YOLO annotation it crops the
    bounding-box region (+10 % padding) and resizes to 224×224.  Crops are
    assigned to ``hummingbird/`` or ``other/`` by annotation class; annotations
    whose class name contains "nobird" / "no_bird" are skipped.  A fresh 80/20
    stratified split is performed on the resulting crop pool.

  Mode B — pre-collected image directories:
    Copies images directly from ``--hummingbird-src`` and ``--other-src``
    directories (e.g. iNaturalist downloads) without any cropping.

Both modes write:

    <out_root>/train/hummingbird/
    <out_root>/train/other/
    <out_root>/val/hummingbird/
    <out_root>/val/other/

which is consumed as-is by HummingbirdClassifier.train_model() via
torchvision.datasets.ImageFolder.

Usage examples
--------------
# Mode A — Roboflow download:
python scripts/prepare_training_data.py \\
    --roboflow-dir data/raw/label-birdfeeder-camera-observations-3 \\
    --out-root data/interim/train_data

# Mode B — pre-collected directories:
python scripts/prepare_training_data.py \\
    --hummingbird-src data/raw/Trochilidae/images \\
    --other-src data/raw/other_birds/Tyrannidae data/raw/other_birds/Corvidae \\
    --out-root data/interim/train_data
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_CROP_SIZE = 224
_PAD_FRACTION = 0.10
_NO_BIRD_KEYWORDS = {"nobird", "no_bird", "no-bird"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split_items(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]  # (train, val)


def _save_pil(pil_img, dest_dir: Path, stem: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pil_img.save(dest_dir / f"{stem}.jpg", quality=95)


# ---------------------------------------------------------------------------
# Mode A — Roboflow YOLOv8/v11 directory
# ---------------------------------------------------------------------------


def _extract_crops_from_image(
    img_path: Path,
    lbl_path: Path,
    hb_ids: set,
    skip_ids: set,
    split_prefix: str,
) -> list[tuple[str, object, str]]:
    """Return list of (class_label, PIL_crop, stem) for one image."""
    from PIL import Image as PILImage

    if not lbl_path.exists():
        return []

    with open(lbl_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []

    try:
        pil_img = PILImage.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"  WARNING: could not open {img_path}: {exc}")
        return []

    img_w, img_h = pil_img.size
    results = []

    for ann_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        if class_id in skip_ids:
            continue

        cx, cy, bw, bh = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        pad_x = (x2 - x1) * _PAD_FRACTION
        pad_y = (y2 - y1) * _PAD_FRACTION
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(img_w, x2 + pad_x)
        y2p = min(img_h, y2 + pad_y)

        crop = pil_img.crop((x1p, y1p, x2p, y2p)).resize(
            (_CROP_SIZE, _CROP_SIZE), PILImage.LANCZOS
        )
        cls_label = "hummingbird" if class_id in hb_ids else "other"
        stem = f"{split_prefix}_{img_path.stem}_ann{ann_idx}"
        results.append((cls_label, crop, stem))

    return results


def _process_roboflow(
    roboflow_dir: Path,
    out_root: Path,
    val_split: float,
    seed: int,
    workers: int = 4,
) -> None:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is required for --roboflow-dir mode: pip install pyyaml"
        )
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_yaml_path = roboflow_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {roboflow_dir}")

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    class_names: list[str] = data_cfg.get("names", [])
    if not class_names:
        raise ValueError(f"No class names found in {data_yaml_path}")

    hb_ids = {i for i, n in enumerate(class_names) if "hummingbird" in n.lower()}
    skip_ids = {
        i
        for i, n in enumerate(class_names)
        if any(kw in n.lower() for kw in _NO_BIRD_KEYWORDS)
    }

    if not hb_ids:
        raise ValueError(
            f"No class named 'hummingbird' found in {data_yaml_path}. "
            f"Classes: {class_names}"
        )
    print(f"Class map: {dict(enumerate(class_names))}")
    print(f"  hummingbird IDs : {hb_ids}")
    print(f"  skipped IDs     : {skip_ids}")

    # Collect all (img_path, lbl_path, split_prefix) tasks
    tasks: list[tuple[Path, Path, str]] = []
    for split_name in ("train", "valid", "test"):
        img_dir = roboflow_dir / split_name / "images"
        lbl_dir = roboflow_dir / split_name / "labels"
        if not img_dir.exists():
            continue
        img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
        print(f"  {split_name}/images: {len(img_paths)} images")
        for img_path in img_paths:
            tasks.append((img_path, lbl_dir / (img_path.stem + ".txt"), split_name))

    print(f"\nExtracting crops with {workers} threads …")
    crops: dict[str, list[tuple[object, str]]] = {"hummingbird": [], "other": []}
    n_done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _extract_crops_from_image,
                img_path,
                lbl_path,
                hb_ids,
                skip_ids,
                split_prefix,
            ): img_path
            for img_path, lbl_path, split_prefix in tasks
        }
        for fut in as_completed(futures):
            n_done += 1
            if n_done % 100 == 0 or n_done == len(tasks):
                print(f"  {n_done}/{len(tasks)} images processed …", flush=True)
            for cls_label, pil_crop, stem in fut.result():
                crops[cls_label].append((pil_crop, stem))

    print(
        f"\nCrops collected — hummingbird: {len(crops['hummingbird'])}, other: {len(crops['other'])}"
    )

    if not crops["hummingbird"]:
        raise SystemExit("No hummingbird crops found — check class names in data.yaml")
    if not crops["other"]:
        raise SystemExit("No 'other' crops found — check class names in data.yaml")

    print("Splitting and saving …")
    for cls_label, items in crops.items():
        train_items, val_items = _split_items(items, val_split, seed)
        for pil_crop, stem in train_items:
            _save_pil(pil_crop, out_root / "train" / cls_label, stem)
        for pil_crop, stem in val_items:
            _save_pil(pil_crop, out_root / "val" / cls_label, stem)

    print("\nFinal counts:")
    for split in ("train", "val"):
        for cls in ("hummingbird", "other"):
            n = len(list((out_root / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls} : {n}")


# ---------------------------------------------------------------------------
# Mode B — pre-collected image directories
# ---------------------------------------------------------------------------


def _collect_images(dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"Source directory not found: {d}")
        images.extend(
            p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS
        )
    return images


def _copy_images(images: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src in images:
        shutil.copy2(src, dest / src.name)


def _process_src_dirs(
    hb_src: Path,
    other_srcs: list[Path],
    out_root: Path,
    val_split: float,
    seed: int,
) -> None:
    hb_images = _collect_images([hb_src])
    other_images = _collect_images(other_srcs)

    if not hb_images:
        raise SystemExit(f"No images found in {hb_src}")
    if not other_images:
        raise SystemExit("No images found in the --other-src directories")

    hb_train, hb_val = _split_items(hb_images, val_split, seed)
    other_train, other_val = _split_items(other_images, val_split, seed)

    _copy_images(hb_train, out_root / "train" / "hummingbird")
    _copy_images(hb_val, out_root / "val" / "hummingbird")
    _copy_images(other_train, out_root / "train" / "other")
    _copy_images(other_val, out_root / "val" / "other")

    print(f"Output root: {out_root.resolve()}")
    print(f"  train/hummingbird : {len(hb_train)}")
    print(f"  train/other       : {len(other_train)}")
    print(f"  val/hummingbird   : {len(hb_val)}")
    print(f"  val/other         : {len(other_val)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode A
    parser.add_argument(
        "--roboflow-dir",
        type=Path,
        default=None,
        help=(
            "Root of a Roboflow YOLOv8/v11 download "
            "(must contain data.yaml and train/valid subdirs). "
            "If provided, --hummingbird-src and --other-src are ignored."
        ),
    )

    # Mode B
    parser.add_argument(
        "--hummingbird-src",
        type=Path,
        default=None,
        help="Directory of hummingbird images (Mode B only)",
    )
    parser.add_argument(
        "--other-src",
        type=Path,
        nargs="+",
        default=None,
        help="One or more directories of non-hummingbird images (Mode B only)",
    )

    # Shared
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/interim/train_data"),
        help="Root output directory (default: data/interim/train_data)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of each class reserved for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel threads for image loading/cropping in Mode A (default: 4)",
    )

    args = parser.parse_args()

    if args.roboflow_dir is not None:
        print(f"Mode A — Roboflow dir: {args.roboflow_dir}")
        _process_roboflow(
            args.roboflow_dir, args.out_root, args.val_split, args.seed, args.workers
        )
    elif args.hummingbird_src is not None and args.other_src is not None:
        print("Mode B — pre-collected image directories")
        _process_src_dirs(
            args.hummingbird_src,
            args.other_src,
            args.out_root,
            args.val_split,
            args.seed,
        )
    else:
        parser.error(
            "Provide either --roboflow-dir (Mode A) "
            "or both --hummingbird-src and --other-src (Mode B)."
        )

    print(f"\nOutput root: {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
