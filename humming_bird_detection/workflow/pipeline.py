"""Pipeline orchestration: process images and write observation CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import pandas as pd

from humming_bird_detection.data.metadata import parse_metadata, extract_margin
from humming_bird_detection.data.preprocessing import crop_and_resize
from humming_bird_detection.models.classifier import HummingbirdClassifier
from humming_bird_detection.models.detector import BirdDetector

logger = logging.getLogger(__name__)


def process_image(
    image_path: str | Path,
    detector: BirdDetector,
    classifier: HummingbirdClassifier,
    cfg: dict,
) -> list[dict]:
    """Process one image: detect birds, classify, extract metadata.

    Returns one record per detected bird (empty list if no birds detected).
    Each record shares the same metadata (filename, date, time, camera) and
    carries per-bird fields: bird_index, n_birds, detection_confidence,
    classification_confidence, and hummingbird label.
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
        records.append(
            {
                "filename": image_path.name,
                "date": metadata.date,
                "time": metadata.time,
                "camera": metadata.camera_id,
                "bird_index": i,
                "n_birds": len(detections),
                "detection_confidence": det.confidence,
                "classification_confidence": conf,
                "hummingbird": label,
            }
        )
    return records


def process_directory(
    input_dir: str | Path,
    output_csv: str | Path,
    detector: BirdDetector,
    classifier: HummingbirdClassifier,
    cfg: dict,
) -> pd.DataFrame:
    """Process all JPEGs in input_dir and write a CSV of observations."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    jpg_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")))
    logger.info("Processing %d images in %s", len(jpg_files), input_dir)

    records = []
    for img_path in jpg_files:
        records.extend(process_image(img_path, detector, classifier, cfg))

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(
        "Processed %d images; wrote %d bird records to %s",
        len(jpg_files),
        len(records),
        output_csv,
    )
    return df
