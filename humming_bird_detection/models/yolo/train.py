"""YOLO training pipeline for hummingbird detection.

Designed for HPC with H100 GPUs. Features:
    - Weights & Biases (wandb) logging for experiment tracking
    - GPU memory monitoring to avoid OOM / SLURM job cancellation
    - Validation every N epochs (default: 5)
    - Early stopping on mAP@0.5
    - Cosine learning rate schedule
    - Mixed precision (AMP) enabled by default
    - Proper evaluation with mAP@0.5, mAP@0.5:0.95, per-class AP
    - Hyperparameter tuning mode (--tune) via Ray Tune
    - Gradient accumulation for effective larger batch sizes

Usage:
    # Standard training on H100
    python -m humming_bird_detection.models.yolo.train --batch-size 32 --img-size 1280

    # Full-resolution training (if preprocessed with --no-resize)
    python -m humming_bird_detection.models.yolo.train --img-size 1920 --batch-size 8

    # Hyperparameter tuning
    python -m humming_bird_detection.models.yolo.train --tune --tune-iterations 30

    # Resume interrupted training
    python -m humming_bird_detection.models.yolo.train --resume

    # Dry-run smoke test (10 images, 2 epochs)
    python -m humming_bird_detection.models.yolo.train --dry-run
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, Optional

import psutil
import torch
from loguru import logger
from ultralytics import YOLO

from humming_bird_detection.configs.paths import (
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from humming_bird_detection.utils.logging import setup_logging

DATA_DIR = PROCESSED_DATA_DIR / "hb_detection" / "hb_ready"
RUNS_DIR = MODELS_DIR / "yolo" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

class GPUMonitor:
    #Background thread that logs GPU memory & utilization at a fixed interval.

    def __init__(
        self,
        interval_sec: int = 60,
        warn_mem_pct: float = 90.0,
        warn_ram_pct: float = 85.0,
    ):
        self.interval = interval_sec
        self.warn_mem_pct = warn_mem_pct
        self.warn_ram_pct = warn_ram_pct
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not torch.cuda.is_available():
            logger.info("No CUDA device — GPU monitoring disabled")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            f"GPU monitor started (interval={self.interval}s, "
            f"warn_gpu_mem>{self.warn_mem_pct}%, warn_ram>{self.warn_ram_pct}%)"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._log_status()
            self._stop.wait(self.interval)

    def _log_status(self) -> None:
        try:
            for i in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                mem_pct = (mem_reserved / mem_total) * 100 if mem_total > 0 else 0

                logger.debug(
                    f"GPU {i}: allocated={mem_alloc:.1f}GB, "
                    f"reserved={mem_reserved:.1f}GB / {mem_total:.1f}GB ({mem_pct:.0f}%)"
                )

                if mem_pct > self.warn_mem_pct:
                    logger.warning(
                        f"⚠️  GPU {i} memory at {mem_pct:.0f}% — risk of OOM! "
                        f"Consider reducing batch size or image size."
                    )

            ram = psutil.virtual_memory()
            ram_pct = ram.percent
            logger.debug(f"System RAM: {ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB ({ram_pct:.0f}%)")
            if ram_pct > self.warn_ram_pct:
                logger.warning(f"⚠️  System RAM at {ram_pct:.0f}% — reduce num_workers if needed.")
        except Exception as exc:
            logger.debug(f"GPU monitor error: {exc}")

    @staticmethod
    def log_gpu_summary() -> None:
        """One-shot GPU info dump for the log header."""
        if not torch.cuda.is_available():
            logger.info("Device: CPU (no CUDA)")
            return
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB | "
                f"Compute {props.major}.{props.minor}"
            )
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

def _setup_wandb(args: argparse.Namespace) -> bool:
    """Initialize Weights & Biases if available and not disabled.
    Returns True if wandb is active.
    """
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("W&B logging DISABLED (--no-wandb)")
        return False

    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb not installed — run `pip install wandb` for experiment tracking")
        os.environ["WANDB_DISABLED"] = "true"
        return False

    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        logger.warning("WANDB_API_KEY not set — W&B logging disabled")
        os.environ["WANDB_DISABLED"] = "true"
        return False

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    logger.info(f"W&B enabled: project={args.wandb_project}")
    return True

def _create_dry_run_dataset(data_yaml: Path, num_samples: int) -> Path:
    """Create a tiny subset dataset for smoke-testing the full pipeline.

    Copies up to `num_samples` images (+labels) from each split into a
    temporary directory and writes a new data.yaml pointing to it.
    Returns the path to the new data.yaml.
    """
    import yaml

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    dry_dir = DATA_DIR.parent / "hb_ready_dryrun"
    if dry_dir.exists():
        shutil.rmtree(dry_dir)

    for split in ("train", "val", "test"):
        src_img = DATA_DIR / split / "images"
        src_lbl = DATA_DIR / split / "labels"
        if not src_img.exists():
            continue

        dst_img = dry_dir / split / "images"
        dst_lbl = dry_dir / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = sorted(src_img.iterdir())[:num_samples]
        for img in images:
            shutil.copy2(img, dst_img / img.name)
            lbl = src_lbl / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)

        logger.info(f"  dry-run {split}: {len(images)} images copied")

    # Write new data.yaml
    dry_yaml = dry_dir / "data.yaml"
    cfg["path"] = str(dry_dir)
    cfg["train"] = "train/images"
    cfg["val"] = "val/images"
    cfg["test"] = "test/images"
    with open(dry_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info(f"Dry-run dataset created at {dry_dir}")
    return dry_yaml

def _log_metrics(results, split: str, class_names: tuple = ("hummingbird", "other_bird")) -> Dict:
    """Extract and log detailed metrics from a YOLO validation result.
    Returns a flat dict of metrics suitable for logging/serialization.
    """
    box = results.box  # DetMetrics.box -> Metric object

    metrics = {
        f"{split}/mAP50": float(box.map50),
        f"{split}/mAP50-95": float(box.map),
        f"{split}/precision": float(box.mp),
        f"{split}/recall": float(box.mr),
    }

    logger.info(f"{'─' * 50}")
    logger.info(f"{split.upper()} METRICS:")
    logger.info(f"  mAP@0.5       = {box.map50:.4f}")
    logger.info(f"  mAP@0.5:0.95  = {box.map:.4f}")
    logger.info(f"  Precision (P) = {box.mp:.4f}")
    logger.info(f"  Recall (R)    = {box.mr:.4f}")

    # Per-class AP
    ap50_per_class = box.ap50
    ap_per_class = box.ap
    if ap50_per_class is not None and len(ap50_per_class) > 0:
        logger.info(f"  Per-class AP@0.5:")
        for i, ap in enumerate(ap50_per_class):
            name = class_names[i] if i < len(class_names) else f"class_{i}"
            logger.info(f"    {name}: AP50={float(ap):.4f}, AP50-95={float(ap_per_class[i]):.4f}")
            metrics[f"{split}/AP50_{name}"] = float(ap)
            metrics[f"{split}/AP50-95_{name}"] = float(ap_per_class[i])

    logger.info(f"{'─' * 50}")
    return metrics

def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    split: str,
    batch_size: int,
    img_size: int,
    device: str,
    class_names: tuple = ("hummingbird", "other_bird"),
    save_dir: Optional[Path] = None,
) -> Dict:
    """Run full evaluation on a split and return metrics dict."""
    logger.info(f"Evaluating {model_path.name} on {split} split...")
    model = YOLO(str(model_path))

    val_kwargs = dict(
        data=str(data_yaml),
        split=split,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        verbose=False,
        plots=True,
    )
    if save_dir:
        val_kwargs["project"] = str(save_dir.parent)
        val_kwargs["name"] = save_dir.name

    results = model.val(**val_kwargs)
    metrics = _log_metrics(results, split, class_names)
    return metrics

def run_tune(args: argparse.Namespace, data_yaml: Path, device: str) -> None:
    """Run Ultralytics hyperparameter tuning with Ray Tune."""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING MODE")
    logger.info(f"  iterations={args.tune_iterations}, epochs per trial={args.tune_epochs}")
    logger.info("  Using Ray Tune (install: pip install ray[tune])")
    logger.info("=" * 60)

    model = YOLO(args.model)
    model.tune(
        data=str(data_yaml),
        epochs=args.tune_epochs,
        iterations=args.tune_iterations,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        project=str(RUNS_DIR),
        name="hb_tune",
        exist_ok=True,
    )
    logger.success("Hyperparameter tuning complete. Check runs/hb_tune/ for results.")

def run_train(args: argparse.Namespace, data_yaml: Path, device: str) -> Path:
    """Execute the training loop and return path to best checkpoint."""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Image size:     {args.img_size}")
    logger.info(f"  Batch size:     {args.batch_size}")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  LR0:            {args.lr0}")
    logger.info(f"  LRf:            {args.lrf}")
    logger.info(f"  Optimizer:      {args.optimizer}")
    logger.info(f"  Val interval:   every {args.val_interval} epochs")
    logger.info(f"  Patience:       {args.patience}")
    logger.info(f"  Device:         {device}")
    logger.info(f"  AMP:            {not args.no_amp}")
    logger.info(f"  Workers:        {args.workers}")
    logger.info(f"  Resume:         {args.resume}")
    logger.info("=" * 60)

    # Start GPU monitor
    monitor = GPUMonitor(
        interval_sec=args.monitor_interval,
        warn_mem_pct=args.warn_gpu_pct,
    )
    monitor.start()

    try:
        if args.resume:
            model = YOLO(args.resume_ckpt)
            logger.info(f"Resuming from: {args.resume_ckpt}")
        else:
            model = YOLO(args.model)

        train_kwargs = dict(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=device,
            project=str(RUNS_DIR),
            name=args.run_name,
            exist_ok=True,
            verbose=True,
            # ── Learning rate
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            cos_lr=True,                      # cosine LR schedule
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # ── Validation & saving
            val=True,
            save=True,
            save_period=args.val_interval,    # save checkpoint every N epochs
            patience=args.patience,           # early stopping patience (0=disabled)
            # ── Mixed precision
            amp=not args.no_amp,
            # ── Data loading
            workers=args.workers,
            # ── Augmentation (YOLO built-in)
            hsv_h=0.015,                      # hue jitter
            hsv_s=0.7,                        # saturation jitter
            hsv_v=0.4,                        # brightness jitter
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,                       # no vertical flip for birds
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            # ── Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            # ── Plots
            plots=True,
            # ── Resume
            resume=args.resume,
        )

        results = model.train(**train_kwargs)
        logger.success("Training finished successfully")
    except torch.cuda.OutOfMemoryError:
        logger.critical(
            "GPU OUT OF MEMORY! Reduce --batch-size or --img-size.\n"
            "  Suggested: halve batch size or reduce img-size from 1280 to 640."
        )
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        monitor.stop()

    # Locate best checkpoint
    run_dir = RUNS_DIR / args.run_name
    weights_dir = run_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    if best_ckpt.exists():
        ckpt = best_ckpt
    elif last_ckpt.exists():
        logger.warning("best.pt not found; falling back to last.pt")
        ckpt = last_ckpt
    else:
        pts = sorted(weights_dir.glob("*.pt"))
        if pts:
            ckpt = pts[-1]
            logger.warning(f"No best/last; using {ckpt.name}")
        else:
            raise FileNotFoundError(f"No checkpoints under {weights_dir}")

    logger.info(f"Best checkpoint: {ckpt}")
    return ckpt

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO for hummingbird detection (H100-optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model
    p.add_argument("--model", type=str, default="yolo26n.pt",
                    help="Pretrained model (yolo26n.pt, yolo26s.pt, yolo26m.pt)")
    p.add_argument("--run-name", type=str, default="hb_detection",
                    help="Experiment run name")

    # ── Training params
    p.add_argument("--epochs", type=int, default=100,
                    help="Total training epochs")
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size (H100 80GB: 32-64 @1280, 128 @640)")
    p.add_argument("--img-size", type=int, default=1280,
                    help="Input image size (1280 recommended for H100)")
    p.add_argument("--lr0", type=float, default=0.01,
                    help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01,
                    help="Final LR as fraction of lr0 (cosine decay target)")
    p.add_argument("--optimizer", type=str, default="AdamW",
                    choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
                    help="Optimizer")

    # ── Validation & early stopping
    p.add_argument("--val-interval", type=int, default=5,
                    help="Run validation + save checkpoint every N epochs")
    p.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience (epochs without mAP improvement; 0=off)")

    # ── Hardware
    p.add_argument("--workers", type=int, default=8,
                    help="Dataloader workers")
    p.add_argument("--no-amp", action="store_true",
                    help="Disable automatic mixed precision")

    # ── GPU monitoring
    p.add_argument("--monitor-interval", type=int, default=60,
                    help="GPU monitoring interval in seconds")
    p.add_argument("--warn-gpu-pct", type=float, default=90.0,
                    help="Warn when GPU memory exceeds this percent")

    # ── W&B
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="hummingbird-detection",
                    help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None,
                    help="W&B team/entity name")

    # ── Resume
    p.add_argument("--resume", action="store_true",
                    help="Resume training from last checkpoint")
    p.add_argument("--resume-ckpt", type=str,
                    default=str(RUNS_DIR / "hb_detection" / "weights" / "last.pt"),
                    help="Checkpoint to resume from")

    # ── Dry-run (smoke test)
    p.add_argument("--dry-run", action="store_true",
                    help="Quick sanity check: train on ~10 images for 2 epochs, then evaluate")
    p.add_argument("--dry-run-samples", type=int, default=10,
                    help="Number of images to use in dry-run mode")

    # ── Hyperparameter tuning
    p.add_argument("--tune", action="store_true",
                    help="Run hyperparameter tuning instead of training")
    p.add_argument("--tune-iterations", type=int, default=30,
                    help="Number of tuning trials")
    p.add_argument("--tune-epochs", type=int, default=30,
                    help="Epochs per tuning trial")

    return p.parse_args()

def main():
    args = get_args()

    # ── Logging (unique file per run) ──
    setup_logging(LOGS_DIR, name="train")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA device detected — training on CPU (very slow!)")

    # Log system info
    GPUMonitor.log_gpu_summary()
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")

    # Data
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run the preprocessing pipeline first:\n"
            f"  python -m humming_bird_detection.preprocessing.preprocessor"
        )

    # W&B
    wandb_active = _setup_wandb(args)

    # ── Dry-run: create a tiny subset dataset for smoke testing
    if args.dry_run:
        data_yaml = _create_dry_run_dataset(data_yaml, args.dry_run_samples)
        args.epochs = min(args.epochs, 2)
        args.patience = 0  # disable early stopping
        args.run_name = args.run_name + "_dryrun"
        logger.info(
            f"DRY-RUN MODE: {args.dry_run_samples} images, "
            f"{args.epochs} epochs, no early stopping"
        )

    if args.tune:
        # ── Hyperparameter tuning mode
        run_tune(args, data_yaml, device)
    else:
        # ── Standard training
        best_ckpt = run_train(args, data_yaml, device)

        # ── Post-training evaluation on val + test
        class_names = ("hummingbird", "other_bird")
        all_metrics = {}

        for split in ("val", "test"):
            split_dir = DATA_DIR / split / "images"
            if not split_dir.exists():
                logger.warning(f"Split '{split}' not found at {split_dir} — skipping evaluation")
                continue
            eval_dir = RUNS_DIR / args.run_name / f"eval_{split}"
            metrics = evaluate_model(
                model_path=best_ckpt,
                data_yaml=data_yaml,
                split=split,
                batch_size=args.batch_size,
                img_size=args.img_size,
                device=device,
                class_names=class_names,
                save_dir=eval_dir,
            )
            all_metrics.update(metrics)

        # Log final metrics to W&B
        if wandb_active:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(all_metrics)
                    logger.info("Final metrics logged to W&B")
            except Exception:
                pass

        # ── Summary
        logger.info("=" * 60)
        logger.info("TRAINING & EVALUATION COMPLETE")
        logger.info(f"  Best model:  {best_ckpt}")
        logger.info(f"  Run dir:     {RUNS_DIR / args.run_name}")
        for k, v in sorted(all_metrics.items()):
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)

        # ── Sample predictions on test images
        test_img_dir = DATA_DIR / "test" / "images"
        if test_img_dir.exists():
            test_images = sorted(test_img_dir.iterdir())[:10]
            if test_images:
                pred_dir = RUNS_DIR / args.run_name / "test_predictions"
                logger.info(f"Saving {len(test_images)} sample predictions to {pred_dir}...")
                best_model = YOLO(str(best_ckpt))
                best_model.predict(
                    source=[str(p) for p in test_images],
                    save=True,
                    project=str(pred_dir.parent),
                    name=pred_dir.name,
                    imgsz=args.img_size,
                    device=device,
                    conf=0.25,
                )
                logger.info(f"Predictions saved to {pred_dir}")

    logger.success("Done.")

if __name__ == "__main__":
    main()
