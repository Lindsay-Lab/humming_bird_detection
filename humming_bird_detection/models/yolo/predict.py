"""Inference script for trained YOLO hummingbird detection models.

Runs prediction on individual images, directories, or video files.
Saves annotated images and a results CSV. Optionally enriches predictions
with timestamp metadata from image_metadata.csv.

Usage:
    # Predict on a directory of images
    python -m humming_bird_detection.models.yolo.predict \\
        --weights models/yolo/runs/hb_detection/weights/best.pt \\
        --source path/to/images/

    # Predict on a single image with custom confidence
    python -m humming_bird_detection.models.yolo.predict \\
        --weights best.pt --source image.jpg --conf 0.3

    # Predict and enrich with timestamp metadata
    python -m humming_bird_detection.models.yolo.predict \\
        --weights best.pt --source path/to/images/ --enrich-metadata
"""
from __future__ import annotations

import csv
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from ultralytics import YOLO

from humming_bird_detection.configs.paths import (
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

# ── Paths ────────────────────────────────────────────────────────────────────
RUNS_DIR = MODELS_DIR / "yolo" / "runs"
METADATA_CSV = PROCESSED_DATA_DIR / "hb_detection" / "image_metadata.csv"

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
)
_log_file = LOGS_DIR / "predict.log"
_log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(_log_file), level="DEBUG", rotation="50 MB")

CLASS_NAMES: Tuple[str, ...] = ("hummingbird", "other_bird")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _load_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load image_metadata.csv into a dict keyed by filename."""
    if not csv_path.exists():
        logger.warning(f"Metadata CSV not found: {csv_path}")
        return {}
    metadata = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["filename"]] = dict(row)
    logger.info(f"Loaded metadata for {len(metadata)} images from {csv_path}")
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLO inference for hummingbird detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--weights", type=str, required=True,
                    help="Path to model weights (.pt)")
    p.add_argument("--source", type=str, required=True,
                    help="Image file, directory, or video to run inference on")
    p.add_argument("--img-size", type=int, default=1280,
                    help="Input image size")
    p.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45,
                    help="IoU threshold for NMS")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: runs/predict/)")
    p.add_argument("--save-txt", action="store_true",
                    help="Save results as YOLO-format .txt labels")
    p.add_argument("--save-csv", action="store_true", default=True,
                    help="Save per-image results CSV")
    p.add_argument("--enrich-metadata", action="store_true",
                    help="Enrich predictions with timestamp metadata from image_metadata.csv")
    p.add_argument("--metadata-csv", type=str, default=None,
                    help="Path to image_metadata.csv (default: auto)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Validate inputs
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata if requested
    metadata = {}
    if args.enrich_metadata:
        csv_path = Path(args.metadata_csv) if args.metadata_csv else METADATA_CSV
        metadata = _load_metadata(csv_path)

    # Run inference
    logger.info(f"Running inference: weights={weights.name}, source={source}, conf={args.conf}")
    model = YOLO(str(weights))

    results = model.predict(
        source=str(source),
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=True,
        save_txt=args.save_txt,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    # Build results CSV
    if args.save_csv:
        csv_path = output_dir / "predictions.csv"
        fieldnames = [
            "filename", "num_detections", "hummingbird_count",
            "other_bird_count", "max_confidence", "detections_json",
        ]
        if metadata:
            fieldnames.extend(["timestamp_datetime", "timestamp_temperature"])

        rows: List[Dict] = []
        for r in results:
            fname = Path(r.path).name
            boxes = r.boxes
            num_dets = len(boxes)
            class_counts = {name: 0 for name in CLASS_NAMES}
            max_conf = 0.0
            det_list = []

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                max_conf = max(max_conf, confidence)
                det_list.append({
                    "class": cls_name,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": [round(c, 1) for c in xyxy],
                })

            row = {
                "filename": fname,
                "num_detections": num_dets,
                "hummingbird_count": class_counts.get("hummingbird", 0),
                "other_bird_count": class_counts.get("other_bird", 0),
                "max_confidence": round(max_conf, 4),
                "detections_json": json.dumps(det_list),
            }

            # Enrich with metadata
            if metadata and fname in metadata:
                meta = metadata[fname]
                row["timestamp_datetime"] = meta.get("timestamp_datetime", "")
                row["timestamp_temperature"] = meta.get("timestamp_temperature", "")

            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Predictions CSV saved to {csv_path} ({len(rows)} images)")

    # Summary
    total_dets = sum(len(r.boxes) for r in results)
    logger.success(
        f"Inference complete: {len(results)} images, {total_dets} total detections. "
        f"Results in {output_dir}"
    )


if __name__ == "__main__":
    main()