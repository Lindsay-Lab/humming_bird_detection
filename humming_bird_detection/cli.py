"""Click CLI entry point for the hummingbird detection pipeline."""

from __future__ import annotations

import logging
import sys

import click

from humming_bird_detection.config import load_config
from humming_bird_detection.models.classifier import build_classifier
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.utils.utils import load_df
from humming_bird_detection.workflow.analysis import (
    export_summary_json,
    generate_plots,
    generate_report,
)
from humming_bird_detection.workflow.pipeline import process_directory
from humming_bird_detection.workflow.postprocessing import (
    aggregate_to_photos,
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
    help="Skip motion-event duplicate-flagging post-processing.",
)
@click.option(
    "--no-suspected-duplicates",
    is_flag=True,
    default=False,
    help="Skip composition-based suspected-duplicate flagging.",
)
@click.option(
    "--no-tiling",
    is_flag=True,
    default=False,
    help=(
        "Disable SAHI tiled inference (fast mode). Only a single full-image "
        "YOLO pass is run per image. Much faster on CPU, but small/far birds "
        "may be missed. Overrides detector.slice_size in config."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "Resume an interrupted run: skip images already recorded in the "
        "<output>.partial.csv sidecar and continue from where it stopped."
    ),
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show a progress bar with ETA (falls back to periodic logs).",
)
@click.option(
    "--report/--no-report",
    default=True,
    show_default=True,
    help="Print a comprehensive analysis report after processing.",
)
@click.option(
    "--json-summary",
    default=None,
    type=click.Path(),
    help="Write a machine-readable JSON summary to this path.",
)
@click.option(
    "--plots-dir",
    default=None,
    type=click.Path(),
    help="Directory to save analysis plots (PNG).",
)
def detect(
    input_dir: str,
    output_csv: str,
    config: str | None,
    detector_model: str | None,
    classifier_model: str | None,
    no_deduplication: bool,
    no_suspected_duplicates: bool,
    no_tiling: bool,
    resume: bool,
    progress: bool,
    report: bool,
    json_summary: str | None,
    plots_dir: str | None,
) -> None:
    """Process a directory of camera photos and write an observation CSV."""
    cfg = load_config(config)

    # Apply CLI overrides.
    if detector_model is not None:
        cfg["detector"]["model"] = detector_model
    if classifier_model is not None:
        cfg["classifier"]["model"] = classifier_model
    if no_tiling:
        # Fast mode: disable SAHI tiling by clearing the slice size.
        cfg["detector"]["slice_size"] = None

    detector = build_detector(cfg)
    classifier = build_classifier(cfg)

    # Strip deduplication / suspected-duplicate config so build_postprocessors
    # won't add them when the user asked to skip.
    if no_deduplication:
        cfg.pop("deduplication", None)
    if no_suspected_duplicates:
        cfg.pop("suspected_duplicates", None)

    df = process_directory(
        input_dir=input_dir,
        output_csv=output_csv,
        detector=detector,
        classifier=classifier,
        cfg=cfg,
        resume=resume,
        progress=progress,
    )
    per_photo_csv = str(output_csv).replace(".csv", "_per_photo.csv")
    click.echo(f"Bird-level CSV : {output_csv} ({len(df)} records)")
    click.echo(f"Per-photo CSV  : {per_photo_csv}")

    if report and not df.empty:
        click.echo(generate_report(df, cfg))

    if json_summary is not None:
        p = export_summary_json(df, json_summary, cfg)
        click.echo(f"JSON summary written to {p}")

    if plots_dir is not None and not df.empty:
        paths = generate_plots(df, plots_dir, cfg)
        click.echo(f"Generated {len(paths)} plot(s) in {plots_dir}")


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


@cli.command()
@click.option(
    "--input-csv",
    required=True,
    type=click.Path(exists=True),
    help="Path to a pipeline output CSV to analyze.",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Path to pipeline.yaml (for threshold annotations).",
)
@click.option(
    "--json-summary",
    default=None,
    type=click.Path(),
    help="Write a machine-readable JSON summary to this path.",
)
@click.option(
    "--plots-dir",
    default=None,
    type=click.Path(),
    help="Directory to save analysis plots (PNG).",
)
@click.option(
    "--per-photo-csv",
    default=None,
    type=click.Path(),
    help="Export a per-photo aggregated CSV to this path.",
)
@click.option(
    "--no-report",
    is_flag=True,
    default=False,
    help="Suppress the terminal report (useful with --json-summary only).",
)
def analyze(
    input_csv: str,
    config: str | None,
    json_summary: str | None,
    plots_dir: str | None,
    per_photo_csv: str | None,
    no_report: bool,
) -> None:
    """Analyze an existing detection CSV and produce a comprehensive report."""
    cfg = load_config(config) if config else None
    df = load_df(input_csv)

    if df.empty:
        click.echo("CSV is empty — nothing to analyze.")
        return

    if not no_report:
        click.echo(generate_report(df, cfg))

    if json_summary is not None:
        p = export_summary_json(df, json_summary, cfg)
        click.echo(f"JSON summary written to {p}")

    if plots_dir is not None:
        paths = generate_plots(df, plots_dir, cfg)
        click.echo(f"Generated {len(paths)} plot(s) in {plots_dir}")

    if per_photo_csv is not None:
        from humming_bird_detection.workflow.postprocessing import aggregate_to_photos

        photos_df = aggregate_to_photos(df)
        photos_df.to_csv(per_photo_csv, index=False)
        click.echo(
            f"Per-photo CSV written to {per_photo_csv} ({len(photos_df)} photos)"
        )
