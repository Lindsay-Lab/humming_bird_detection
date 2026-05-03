"""Fine-tune the EfficientNetV2-S hummingbird classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import wandb

from humming_bird_detection.models.classifier import HummingbirdClassifier
from humming_bird_detection.project_paths import project_paths
from humming_bird_detection.utils.wandb_fetcher import save_all_wandb_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune hummingbird classifier")
    p.add_argument(
        "--train-dir",
        default=str(project_paths.data.interim / "train_data" / "train"),
        help="ImageFolder-style train directory (hummingbird/ and other/ subdirs)",
    )
    p.add_argument(
        "--val-dir",
        default=str(project_paths.data.interim / "train_data" / "val"),
        help="ImageFolder-style validation directory",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and PyTorch (set to -1 to skip seeding)",
    )
    p.add_argument(
        "--output-path",
        default=str(project_paths.models / f"hummingbird_classifier_{_RUN_ID}.pt"),
        help="Where to save the best checkpoint (run ID is embedded in the default name)",
    )
    p.add_argument(
        "--wandb-project",
        default="hummingbird-detection",
        help="Weights & Biases project name",
    )
    p.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity (team or username); uses your default when omitted",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── reproducibility ──────────────────────────────────────────────────
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        logger.info("Random seed set to %d", args.seed)
    else:
        logger.info("Seeding skipped (--seed -1)")

    # ── W&B init ────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=_RUN_ID,
            config={
                "run_id": _RUN_ID,
                "model": "efficientnet_v2_s",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.early_stopping_patience,
                "seed": args.seed,
                "train_dir": str(args.train_dir),
                "val_dir": str(args.val_dir),
                "device": device,
            },
            tags=["classifier", "efficientnet_v2_s", "finetune"],
        )
        logger.info("W&B run: %s", wandb_run.url)

    classifier = HummingbirdClassifier(device=device)

    results = classifier.train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_path=args.output_path,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        wandb_run=wandb_run,
    )

    logger.info(
        "Training complete — best val loss: %.4f at epoch %d",
        results["best_val_loss"],
        results["best_epoch"],
    )
    logger.info("Model saved to: %s", args.output_path)

    # ── fetch WandB data (local → API): history, system metrics, metadata,
    #    summary, and output log — saved alongside the checkpoint ────────────
    if wandb_run is not None:
        saved = save_all_wandb_data(wandb_run, args.output_path)
        logger.info(
            "WandB data saved: %s",
            {k: (str(v) if v else "unavailable") for k, v in saved.items()},
        )

    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"hummingbird_classifier_{_RUN_ID}",
            type="model",
            metadata={
                "best_val_loss": results["best_val_loss"],
                "best_epoch": results["best_epoch"],
            },
        )
        artifact.add_file(str(args.output_path))
        wandb_run.log_artifact(artifact)
        wandb_run.summary.update(
            {
                "best_val_loss": results["best_val_loss"],
                "best_epoch": results["best_epoch"],
            }
        )
        wandb_run.finish()
        logger.info("W&B run finished: %s", wandb_run.url)

    # ── save metadata for provenance (wandb enriches this later) ─────────────
    def _git_hash() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    config = {
        "run_id": _RUN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _git_hash(),
        "wandb": {
            "run_id": wandb_run.id if wandb_run is not None else None,
            "run_url": wandb_run.url if wandb_run is not None else None,
            "project": args.wandb_project if not args.no_wandb else None,
            "entity": args.wandb_entity if not args.no_wandb else None,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
        },
        "data": {
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
        },
        "results": {
            "best_val_loss": results["best_val_loss"],
            "best_epoch": results["best_epoch"],
        },
        "output_path": str(args.output_path),
    }
    out_dir = Path(args.output_path).parent
    stem = Path(args.output_path).stem
    config_path = out_dir / f"{stem}_metadata.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Metadata saved to: %s", config_path)


if __name__ == "__main__":
    main()
