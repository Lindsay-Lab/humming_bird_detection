"""Click CLI entry point for the hummingbird detection pipeline."""

from __future__ import annotations

import logging
import sys

import click

from humming_bird_detection.config import load_config
from humming_bird_detection.models.classifier import build_classifier
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.workflow.pipeline import process_directory
from humming_bird_detection.workflow.postprocessing import (
    apply_postprocessors,
    build_postprocessors,
)

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Logging verbosity.",
)
def cli(log_level: str) -> None:
    """Hummingbird Detection Pipeline CLI."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory of JPEG photos to process.",
)
@click.option(
    "--output-csv",
    required=True,
    type=click.Path(),
    help="Path to write results CSV.",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Path to pipeline.yaml. Defaults to data/interim/pipeline.yaml.",
)
@click.option(
    "--detector-model",
    default=None,
    help="Override detector model filename from config (e.g. yolo11s.pt).",
)
@click.option(
    "--classifier-model",
    default=None,
    type=click.Path(),
    help="Override classifier model path from config.",
)
@click.option(
    "--no-deduplication",
    is_flag=True,
    default=False,
    help="Skip duplicate-flagging post-processing.",
)
def detect(
    input_dir: str,
    output_csv: str,
    config: str | None,
    detector_model: str | None,
    classifier_model: str | None,
    no_deduplication: bool,
) -> None:
    """Process a directory of camera photos and write an observation CSV."""
    cfg = load_config(config)

    # Apply CLI overrides.
    if detector_model is not None:
        cfg["detector"]["model"] = detector_model
    if classifier_model is not None:
        cfg["classifier"]["model"] = classifier_model

    detector = build_detector(cfg)
    classifier = build_classifier(cfg)

    # Strip deduplication config so build_postprocessors won't add it.
    if no_deduplication:
        cfg.pop("deduplication", None)

    df = process_directory(
        input_dir=input_dir,
        output_csv=output_csv,
        detector=detector,
        classifier=classifier,
        cfg=cfg,
    )
    click.echo(f"Wrote {len(df)} records to {output_csv}")


@cli.command()
@click.option(
    "--train-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Training data directory (must contain hummingbird/ and other/ subdirs).",
)
@click.option(
    "--val-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Validation data directory (must contain hummingbird/ and other/ subdirs).",
)
@click.option(
    "--output-model",
    default=None,
    type=click.Path(),
    help="Path to save trained weights (.pt). Defaults to models/hummingbird_classifier.pt.",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Path to pipeline.yaml. Defaults to data/interim/pipeline.yaml.",
)
@click.option("--epochs", default=None, type=int, help="Override training epochs.")
@click.option("--lr", default=None, type=float, help="Override learning rate.")
@click.option("--batch-size", default=None, type=int, help="Override batch size.")
def train(
    train_dir: str,
    val_dir: str,
    output_model: str | None,
    config: str | None,
    epochs: int | None,
    lr: float | None,
    batch_size: int | None,
) -> None:
    """Fine-tune the hummingbird classifier on labeled data."""
    cfg = load_config(config)
    train_cfg = cfg.get("training", {})

    # CLI options override config values; None falls back to config.
    resolved_epochs = epochs if epochs is not None else train_cfg.get("epochs", 30)
    resolved_lr = lr if lr is not None else train_cfg.get("learning_rate", 3e-4)
    resolved_batch = (
        batch_size if batch_size is not None else train_cfg.get("batch_size", 32)
    )

    classifier = build_classifier(cfg)
    result = classifier.train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=resolved_epochs,
        lr=resolved_lr,
        batch_size=resolved_batch,
        output_path=output_model,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
    )
    click.echo(
        f"Training complete. Best val loss: {result['best_val_loss']:.4f} "
        f"at epoch {result['best_epoch']}."
    )
