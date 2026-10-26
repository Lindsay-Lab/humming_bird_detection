"""Pipeline orchestration: process images and write observation CSV."""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from humming_bird_detection.data.metadata import parse_metadata, extract_margin
from humming_bird_detection.data.preprocessing import crop_and_resize
from humming_bird_detection.models.classifier import HummingbirdClassifier
from humming_bird_detection.models.detector import BirdDetector
from humming_bird_detection.workflow.postprocessing import (
    apply_postprocessors,
    build_postprocessors,
)

logger = logging.getLogger(__name__)

# Column order for the incremental bird-level record CSV.  Kept explicit so
# that a partially written file (from an interrupted run) can be re-read and
# resumed deterministically.
_RECORD_FIELDS = [
    "filename",
    "date",
    "time",
    "temperature",
    "camera",
    "bird_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "detection_confidence",
    "classification_confidence",
    "is_hummingbird",
]


def _iter_progress(iterable, total: int, desc: str):
    """Yield items from *iterable* with a tqdm progress bar when available.

    Falls back to periodic INFO logging (every 10 items) if tqdm is not
    installed or stderr is not a TTY, so long runs are never silent.
    """
    try:
        from tqdm import tqdm  # noqa: PLC0415

        yield from tqdm(iterable, total=total, desc=desc, unit="img")
        return
    except Exception:  # pragma: no cover - tqdm always present via deps
        pass

    start = time.monotonic()
    for i, item in enumerate(iterable, start=1):
        yield item
        if i % 10 == 0 or i == total:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed else 0.0
            remaining = (total - i) / rate if rate else float("nan")
            logger.info(
                "%s: %d/%d images (%.1f%%, %.2fs/img, ~%.0fs remaining)",
                desc,
                i,
                total,
                100.0 * i / total if total else 0.0,
                1.0 / rate if rate else float("nan"),
                remaining,
            )


def process_image(
    image_path: str | Path,
    detector: BirdDetector,
    classifier: HummingbirdClassifier,
    cfg: dict,
) -> list[dict]:
    """Process one image: detect birds, classify, extract metadata.

    Returns one record per detected bird (empty list if no birds detected).
    Each record shares the same metadata (filename, date, time, temperature, camera) and
    carries per-bird fields: bird_index, bbox_x1/y1/x2/y2 (pixel coordinates of
    the detection in the original image), detection_confidence,
    classification_confidence, and is_hummingbird label.
    """
    image_path = Path(image_path)
    margin_px: int = cfg["image"]["margin_px"]

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Could not read image: %s", image_path)
        return []

    margin = extract_margin(image, margin_px)
    inference_image = image[:-margin_px, :, :]

    detections = detector.detect(inference_image)
    if not detections:
        return []

    crops = [
        crop_and_resize(
            inference_image,
            d.bbox,
            pad_fraction=cfg["preprocessing"]["pad_fraction"],
            output_size=cfg["classifier"]["image_size"],
        )
        for d in detections
    ]

    predictions = classifier.predict(crops)
    metadata = parse_metadata(image_path.name, margin)

    records = []
    for i, (det, (label, conf)) in enumerate(zip(detections, predictions)):
        x1, y1, x2, y2 = det.bbox
        records.append(
            {
                "filename": image_path.name,
                "date": metadata.date,
                "time": metadata.time,
                "temperature": metadata.temperature,
                "camera": metadata.camera_id,
                "bird_index": i,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "detection_confidence": det.confidence,
                "classification_confidence": conf,
                "is_hummingbird": bool(label),
            }
        )
    return records


def _list_input_images(input_dir: Path) -> list[Path]:
    """Return the sorted, de-duplicated list of JPEGs in *input_dir*.

    Case-insensitive filesystems can surface the same file twice via both the
    ``*.jpg`` and ``*.JPG`` globs; we de-duplicate by resolved path so each
    image is processed exactly once.
    """
    seen: dict[str, Path] = {}
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        for p in input_dir.glob(pattern):
            seen.setdefault(str(p.resolve()), p)
    return sorted(seen.values(), key=lambda p: p.name)


def process_directory(
    input_dir: str | Path,
    output_csv: str | Path | None,
    detector: BirdDetector,
    classifier: HummingbirdClassifier,
    cfg: dict,
    resume: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Process all JPEGs in input_dir and write a CSV of observations.

    If output_csv is None, the file is written to
    ``reports/<input_dir_name>_<YYYYMMDD_HHMMSS>.csv``.

    A companion per-photo CSV is written alongside the bird-level CSV when
    *output_csv* is not None — its path is derived by inserting ``_per_photo``
    before the ``.csv`` extension.

    Robustness features
    -------------------
    Detection on CPU is expensive (SAHI tiled inference on a blank image can
    take tens of seconds), so a full directory may take many minutes.  To make
    long runs safe and observable:

    - **Incremental writes.** Every bird-level record is appended to a raw
      ``<output>.partial.csv`` sidecar immediately after each image is
      processed.  If the run is interrupted, completed work is preserved.
    - **Resume.** When *resume* is True and a ``.partial.csv`` sidecar exists,
      images whose filenames already appear in it are skipped and their
      records reloaded, so a killed run can be restarted from where it stopped.
    - **Progress.** When *progress* is True a tqdm progress bar (with ETA) is
      shown; otherwise, or without a TTY, progress is logged periodically.

    Parameters
    ----------
    resume:
        Skip images already recorded in the ``.partial.csv`` sidecar.
    progress:
        Display a progress bar / periodic progress logs.

    Returns
    -------
    pd.DataFrame
        Bird-level DataFrame (one row per bird detection).  The attributes
        ``df.attrs[\"total_images\"]`` and ``df.attrs[\"blank_images\"]`` carry
        the total number of JPEG files processed and the number with no birds.
    """
    input_dir = Path(input_dir)
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = Path("reports") / f"{input_dir.name}_{timestamp}.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    partial_csv = output_csv.parent / (output_csv.stem + ".partial.csv")

    jpg_files = _list_input_images(input_dir)
    total_images = len(jpg_files)

    # --- Resume: reload records from a prior partial run ---
    prior_records: list[dict] = []
    processed_filenames: set[str] = set()
    if resume and partial_csv.exists():
        prior_df = pd.read_csv(partial_csv)
        prior_records = prior_df.to_dict("records")
        processed_filenames = set(prior_df["filename"].unique())
        # Blank images are recorded as a sentinel row with bird_index == -1.
        prior_records = [r for r in prior_records if r.get("bird_index", 0) != -1]
        logger.info(
            "Resuming: %d images already processed (from %s)",
            len(processed_filenames),
            partial_csv,
        )

    pending = [p for p in jpg_files if p.name not in processed_filenames]
    logger.info(
        "Processing %d images in %s (%d already done, %d pending)",
        total_images,
        input_dir,
        len(processed_filenames),
        len(pending),
    )

    # --- Open the partial sidecar for incremental appends ---
    write_header = not (resume and partial_csv.exists())
    fh = partial_csv.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=_RECORD_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()

    records: list[dict] = list(prior_records)
    iterator = pending
    if progress:
        iterator = _iter_progress(pending, total=len(pending), desc="Detecting")

    try:
        for img_path in iterator:
            img_records = process_image(img_path, detector, classifier, cfg)
            if img_records:
                for rec in img_records:
                    writer.writerow({k: rec.get(k) for k in _RECORD_FIELDS})
                records.extend(img_records)
            else:
                # Sentinel row so blanks are counted correctly on resume.
                writer.writerow({"filename": img_path.name, "bird_index": -1})
            fh.flush()
    finally:
        fh.close()

    # Reconstruct blank count from the full image set vs. images with records.
    filenames_with_birds = {r["filename"] for r in records}
    blank_count = total_images - len(filenames_with_birds)

    df = pd.DataFrame(records)
    df = apply_postprocessors(df, build_postprocessors(cfg))
    df.attrs["total_images"] = total_images
    df.attrs["blank_images"] = blank_count

    df.to_csv(output_csv, index=False)
    logger.info(
        "Processed %d images; wrote %d bird records to %s (%d blank images)",
        total_images,
        len(records),
        output_csv,
        blank_count,
    )

    # --- Per-photo CSV ---
    from humming_bird_detection.workflow.postprocessing import aggregate_to_photos

    photos_df = aggregate_to_photos(df)
    photo_csv = output_csv.parent / output_csv.name.replace(".csv", "_per_photo.csv")
    photos_df.to_csv(photo_csv, index=False)
    logger.info("Per-photo summary written to %s (%d photos)", photo_csv, len(photos_df))

    # Success — remove the partial sidecar now that final outputs are written.
    try:
        partial_csv.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        logger.debug("Could not remove partial file %s", partial_csv)

    return df
