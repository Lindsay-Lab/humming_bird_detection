"""Standalone evaluation script for trained YOLO hummingbird detection models.

Produces:
    - mAP@0.5, mAP@0.5:0.95, precision, recall (overall + per-class)
    - Confusion matrix (saved as image)
    - Precision–Recall curves (saved as image)
    - F1–Confidence curve
    - Per-image inference results CSV
    - Optional W&B logging

Usage:
    # Evaluate best checkpoint on test split
    python -m humming_bird_detection.models.yolo.eval \\
        --weights models/yolo/runs/hb_detection/weights/best.pt \\
        --split test

    # Evaluate on val split with custom confidence threshold
    python -m humming_bird_detection.models.yolo.eval \\
        --weights models/yolo/runs/hb_detection/weights/best.pt \\
        --split val --conf 0.25
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
DATA_DIR = PROCESSED_DATA_DIR / "hb_detection" / "hb_ready"
RUNS_DIR = MODELS_DIR / "yolo" / "runs"

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
)
_log_file = LOGS_DIR / "eval.log"
_log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(_log_file), level="DEBUG", rotation="50 MB")

CLASS_NAMES: Tuple[str, ...] = ("hummingbird", "other_bird")


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(results, split: str) -> Dict[str, float]:
    """Extract a flat metrics dict from YOLO validation results."""
    box = results.box

    metrics: Dict[str, float] = {
        f"{split}/mAP50": float(box.map50),
        f"{split}/mAP50-95": float(box.map),
        f"{split}/precision": float(box.mp),
        f"{split}/recall": float(box.mr),
    }

    # Per-class
    ap50 = box.ap50
    ap = box.ap
    if ap50 is not None and len(ap50) > 0:
        for i, (a50, a) in enumerate(zip(ap50, ap)):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            metrics[f"{split}/AP50_{name}"] = float(a50)
            metrics[f"{split}/AP50-95_{name}"] = float(a)

    return metrics


def print_metrics_table(metrics: Dict[str, float]) -> None:
    """Pretty-print a metrics dict."""
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    # Group by split prefix
    for key in sorted(metrics.keys()):
        logger.info(f"  {key:40s} = {metrics[key]:.4f}")

    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image inference results
# ─────────────────────────────────────────────────────────────────────────────

def run_per_image_eval(
    model: YOLO,
    img_dir: Path,
    output_csv: Path,
    img_size: int,
    conf: float,
    iou: float,
    device: str,
) -> int:
    """Run inference on each image and write per-image results to CSV.

    CSV columns: filename, num_detections, hummingbird_count, other_bird_count,
                 max_confidence, detections_json
    Returns the number of images processed.
    """
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not images:
        logger.warning(f"No images found in {img_dir}")
        return 0

    logger.info(f"Running per-image inference on {len(images)} images...")

    rows: List[Dict] = []
    for img_path in images:
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        r = results[0]
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

        rows.append({
            "filename": img_path.name,
            "num_detections": num_dets,
            "hummingbird_count": class_counts.get("hummingbird", 0),
            "other_bird_count": class_counts.get("other_bird", 0),
            "max_confidence": round(max_conf, 4),
            "detections_json": json.dumps(det_list),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "num_detections", "hummingbird_count",
            "other_bird_count", "max_confidence", "detections_json",
        ])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Per-image results saved to {output_csv} ({len(rows)} images)")
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained YOLO hummingbird detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--weights", type=str, required=True,
                    help="Path to model weights (.pt)")
    p.add_argument("--data", type=str, default=None,
                    help="Path to data.yaml (default: auto from paths.py)")
    p.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"],
                    help="Dataset split to evaluate on")
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for validation")
    p.add_argument("--img-size", type=int, default=1280,
                    help="Image size")
    p.add_argument("--conf", type=float, default=0.001,
                    help="Confidence threshold for mAP computation (low for proper mAP)")
    p.add_argument("--iou", type=float, default=0.6,
                    help="IoU threshold for NMS")
    p.add_argument("--per-image", action="store_true",
                    help="Also run per-image inference and save CSV")
    p.add_argument("--per-image-conf", type=float, default=0.25,
                    help="Confidence threshold for per-image inference CSV")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for results (default: auto)")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging")
    p.add_argument("--wandb-project", type=str, default="hummingbird-detection",
                    help="W&B project name")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Paths
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    data_yaml = Path(args.data) if args.data else DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        RUNS_DIR / "eval" / f"{weights.stem}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Weights:    {weights}")
    logger.info(f"data.yaml:  {data_yaml}")
    logger.info(f"Split:      {args.split}")
    logger.info(f"Output:     {output_dir}")

    # ── Standard YOLO validation (mAP, confusion matrix, PR curves)
    logger.info("Running YOLO validation...")
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        batch=args.batch_size,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=True,
        plots=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    metrics = extract_metrics(results, args.split)
    print_metrics_table(metrics)

    # Save metrics as JSON
    metrics_json = output_dir / "metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics JSON saved to {metrics_json}")

    # ── Per-image inference CSV
    if args.per_image:
        split_img_dir = DATA_DIR / args.split / "images"
        if split_img_dir.exists():
            per_img_csv = output_dir / f"per_image_{args.split}.csv"
            run_per_image_eval(
                model=model,
                img_dir=split_img_dir,
                output_csv=per_img_csv,
                img_size=args.img_size,
                conf=args.per_image_conf,
                iou=args.iou,
                device=device,
            )
        else:
            logger.warning(f"Split image dir not found: {split_img_dir}")

    # ── W&B
    if not args.no_wandb:
        try:
            import wandb
            wandb_key = os.getenv("WANDB_API_KEY")
            if wandb_key:
                wandb.init(
                    project=args.wandb_project,
                    name=f"eval_{weights.stem}_{args.split}",
                    config=vars(args),
                )
                wandb.log(metrics)
                # Log plots if they exist
                for plot_name in ("confusion_matrix.png", "PR_curve.png", "F1_curve.png"):
                    plot_path = output_dir / plot_name
                    if plot_path.exists():
                        wandb.log({plot_name.replace(".png", ""): wandb.Image(str(plot_path))})
                wandb.finish()
                logger.info("Evaluation metrics logged to W&B")
        except ImportError:
            logger.debug("wandb not installed — skipping W&B logging")
        except Exception as exc:
            logger.debug(f"W&B logging failed: {exc}")

    logger.success(f"Evaluation complete. Results in {output_dir}")


if __name__ == "__main__":
    main()