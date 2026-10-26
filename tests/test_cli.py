"""Tests for the hummingbird detection CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from humming_bird_detection.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------


def test_detect_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["detect", "--help"])
    assert result.exit_code == 0
    assert "--input-dir" in result.output


def test_train_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--train-dir" in result.output


def test_top_level_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "detect" in result.output
    assert "train" in result.output


# ---------------------------------------------------------------------------
# detect command — functional invocation
# ---------------------------------------------------------------------------


def test_detect_invocation(runner: CliRunner, tmp_path: Path) -> None:
    """detect runs end-to-end with mocked pipeline internals."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    output_csv = tmp_path / "out.csv"

    mock_df = pd.DataFrame(
        [
            {
                "filename": "a.jpg",
                "is_hummingbird": True,
                "detection_confidence": 0.85,
                "classification_confidence": 0.9,
            }
        ]
    )

    with (
        patch("humming_bird_detection.cli.load_config") as mock_cfg,
        patch("humming_bird_detection.cli.build_detector") as mock_det,
        patch("humming_bird_detection.cli.build_classifier") as mock_cls,
        patch("humming_bird_detection.cli.process_directory", return_value=mock_df),
    ):
        mock_cfg.return_value = {
            "image": {"margin_px": 272},
            "detector": {
                "model": "yolo11n.pt",
                "bird_class_id": 14,
                "confidence_threshold": 0.3,
            },
            "classifier": {
                "model": "hummingbird_classifier.pt",
                "image_size": 224,
                "confidence_threshold": 0.5,
            },
            "preprocessing": {"pad_fraction": 0.1},
        }
        result = runner.invoke(
            cli,
            ["detect", "--input-dir", str(input_dir), "--output-csv", str(output_csv)],
        )

    assert result.exit_code == 0, result.output
    assert "1 records" in result.output


def test_detect_no_deduplication_flag(runner: CliRunner, tmp_path: Path) -> None:
    """--no-deduplication removes deduplication key from cfg before processing."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    output_csv = tmp_path / "out.csv"

    captured_cfg: list[dict] = []

    def _capture_cfg(input_dir, output_csv, detector, classifier, cfg, **kwargs):
        captured_cfg.append(dict(cfg))
        return pd.DataFrame()

    base_cfg = {
        "image": {"margin_px": 272},
        "detector": {
            "model": "yolo11n.pt",
            "bird_class_id": 14,
            "confidence_threshold": 0.3,
        },
        "classifier": {
            "model": "hummingbird_classifier.pt",
            "image_size": 224,
            "confidence_threshold": 0.5,
        },
        "preprocessing": {"pad_fraction": 0.1},
        "deduplication": {"window_seconds": 18},
    }

    with (
        patch("humming_bird_detection.cli.load_config", return_value=base_cfg),
        patch("humming_bird_detection.cli.build_detector"),
        patch("humming_bird_detection.cli.build_classifier"),
        patch(
            "humming_bird_detection.cli.process_directory", side_effect=_capture_cfg
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "detect",
                "--input-dir",
                str(input_dir),
                "--output-csv",
                str(output_csv),
                "--no-deduplication",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "deduplication" not in captured_cfg[0]


def test_detect_model_override(runner: CliRunner, tmp_path: Path) -> None:
    """--detector-model overrides the model name in cfg."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    output_csv = tmp_path / "out.csv"

    captured_cfg: list[dict] = []

    def _capture(input_dir, output_csv, detector, classifier, cfg, **kwargs):
        captured_cfg.append(dict(cfg))
        return pd.DataFrame()

    base_cfg = {
        "image": {"margin_px": 272},
        "detector": {
            "model": "yolo11n.pt",
            "bird_class_id": 14,
            "confidence_threshold": 0.3,
        },
        "classifier": {
            "model": "hummingbird_classifier.pt",
            "image_size": 224,
            "confidence_threshold": 0.5,
        },
        "preprocessing": {"pad_fraction": 0.1},
    }

    with (
        patch("humming_bird_detection.cli.load_config", return_value=base_cfg),
        patch("humming_bird_detection.cli.build_detector"),
        patch("humming_bird_detection.cli.build_classifier"),
        patch("humming_bird_detection.cli.process_directory", side_effect=_capture),
    ):
        result = runner.invoke(
            cli,
            [
                "detect",
                "--input-dir",
                str(input_dir),
                "--output-csv",
                str(output_csv),
                "--detector-model",
                "yolo11s.pt",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_cfg[0]["detector"]["model"] == "yolo11s.pt"


# ---------------------------------------------------------------------------
# train command — functional invocation
# ---------------------------------------------------------------------------


def test_train_invocation(runner: CliRunner, tmp_path: Path) -> None:
    """train runs end-to-end with mocked classifier.train_model."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    mock_classifier = MagicMock()
    mock_classifier.train_model.return_value = {"best_val_loss": 0.05, "best_epoch": 4}

    with (
        patch("humming_bird_detection.cli.load_config") as mock_cfg,
        patch(
            "humming_bird_detection.cli.build_classifier", return_value=mock_classifier
        ),
    ):
        mock_cfg.return_value = {
            "image": {"margin_px": 272},
            "detector": {
                "model": "yolo11n.pt",
                "bird_class_id": 14,
                "confidence_threshold": 0.3,
            },
            "classifier": {
                "model": "hummingbird_classifier.pt",
                "image_size": 224,
                "confidence_threshold": 0.5,
            },
            "preprocessing": {"pad_fraction": 0.1},
            "training": {
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "early_stopping_patience": 3,
            },
        }
        result = runner.invoke(
            cli,
            ["train", "--train-dir", str(train_dir), "--val-dir", str(val_dir)],
        )

    assert result.exit_code == 0, result.output
    assert "0.0500" in result.output
    assert "epoch 4" in result.output


def test_train_cli_overrides_config(runner: CliRunner, tmp_path: Path) -> None:
    """--epochs, --lr, --batch-size override config values."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    mock_classifier = MagicMock()
    mock_classifier.train_model.return_value = {"best_val_loss": 0.1, "best_epoch": 1}
    captured: list[dict] = []

    def _capture(**kwargs):
        captured.append(kwargs)
        return {"best_val_loss": 0.1, "best_epoch": 1}

    mock_classifier.train_model.side_effect = _capture

    with (
        patch("humming_bird_detection.cli.load_config") as mock_cfg,
        patch(
            "humming_bird_detection.cli.build_classifier", return_value=mock_classifier
        ),
    ):
        mock_cfg.return_value = {
            "image": {"margin_px": 272},
            "detector": {
                "model": "yolo11n.pt",
                "bird_class_id": 14,
                "confidence_threshold": 0.3,
            },
            "classifier": {
                "model": "hummingbird_classifier.pt",
                "image_size": 224,
                "confidence_threshold": 0.5,
            },
            "preprocessing": {"pad_fraction": 0.1},
            "training": {
                "epochs": 30,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "early_stopping_patience": 5,
            },
        }
        result = runner.invoke(
            cli,
            [
                "train",
                "--train-dir",
                str(train_dir),
                "--val-dir",
                str(val_dir),
                "--epochs",
                "2",
                "--lr",
                "0.001",
                "--batch-size",
                "4",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured[0]["epochs"] == 2
    assert captured[0]["lr"] == pytest.approx(0.001)
    assert captured[0]["batch_size"] == 4
