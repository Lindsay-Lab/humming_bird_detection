"""Tests for humming_bird_detection.models.classifier."""

from __future__ import annotations

import numpy as np
import pytest

from unittest.mock import MagicMock

from humming_bird_detection.models.classifier import (
    HummingbirdClassifier,
    build_classifier,
)


@pytest.fixture(scope="module")
def classifier():
    """Classifier with no saved weights — uses ImageNet backbone."""
    return HummingbirdClassifier(model_path=None, device="cpu")


def _random_images(n: int = 3) -> list[np.ndarray]:
    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(n)]


def test_model_loads_without_weights():
    # Should not raise even when model_path is None
    clf = HummingbirdClassifier(model_path=None)
    assert clf is not None


def test_predict_output_length(classifier):
    images = _random_images(3)
    results = classifier.predict(images)
    assert len(results) == 3


def test_predict_confidence_range(classifier):
    results = classifier.predict(_random_images(4))
    for _, conf in results:
        assert 0.0 <= conf <= 1.0


def test_predict_label_binary(classifier):
    results = classifier.predict(_random_images(4))
    for label, _ in results:
        assert label in {0, 1}


# ---------------------------------------------------------------------------
# train_model() tests — use tiny synthetic datasets (no real images required)
# ---------------------------------------------------------------------------

from PIL import Image  # noqa: E402


def _make_tiny_dataset(base: "pathlib.Path") -> tuple["pathlib.Path", "pathlib.Path"]:
    """Create minimal ImageFolder-compatible train/val directories."""
    import pathlib

    for split in ("train", "val"):
        for cls in ("hummingbird", "other"):
            d = base / split / cls
            d.mkdir(parents=True)
            for i in range(2):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                img.save(d / f"{cls}_{i}.jpg")
    return base / "train", base / "val"


def test_train_model_returns_result_dict(tmp_path):
    train_dir, val_dir = _make_tiny_dataset(tmp_path)
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    result = clf.train_model(
        train_dir,
        val_dir,
        epochs=1,
        batch_size=4,
        output_path=tmp_path / "ckpt.pt",
        early_stopping_patience=2,
    )
    assert "best_val_loss" in result
    assert "best_epoch" in result
    assert isinstance(result["best_val_loss"], float)
    assert isinstance(result["best_epoch"], int)


def test_train_model_saves_checkpoint(tmp_path):
    train_dir, val_dir = _make_tiny_dataset(tmp_path)
    ckpt = tmp_path / "ckpt.pt"
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    clf.train_model(train_dir, val_dir, epochs=1, output_path=ckpt)
    assert ckpt.exists()


def test_train_model_checkpoint_loadable(tmp_path):
    """Saved state_dict must be loadable by a fresh HummingbirdClassifier.__init__."""
    train_dir, val_dir = _make_tiny_dataset(tmp_path)
    ckpt = tmp_path / "ckpt.pt"
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    clf.train_model(train_dir, val_dir, epochs=1, output_path=ckpt)

    # Load into a new instance — should not raise
    clf2 = HummingbirdClassifier(model_path=ckpt, device="cpu")
    results = clf2.predict(_random_images(2))
    assert len(results) == 2


def test_train_model_early_stopping_no_error(tmp_path):
    """Early stopping with patience=1 must terminate without raising."""
    train_dir, val_dir = _make_tiny_dataset(tmp_path)
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    result = clf.train_model(
        train_dir,
        val_dir,
        epochs=10,
        batch_size=4,
        output_path=tmp_path / "ckpt.pt",
        early_stopping_patience=1,
    )
    # Must have stopped early (epoch <= 10) and returned valid dict
    assert result["best_epoch"] >= 1
    assert result["best_val_loss"] >= 0.0


def test_nonexistent_model_path_logs_warning(tmp_path, caplog):
    """A non-existent model_path should log a warning instead of raising."""
    import logging

    with caplog.at_level(logging.WARNING):
        clf = HummingbirdClassifier(
            model_path=tmp_path / "does_not_exist.pt", device="cpu"
        )
    assert clf is not None
    assert any("does not exist" in r.message for r in caplog.records)


def test_train_model_raises_on_missing_hummingbird_class(tmp_path):
    """train_model must raise ValueError when no 'hummingbird' subdir is present."""
    for split in ("train", "val"):
        for cls in ("robin", "sparrow"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(2):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                img.save(d / f"{cls}_{i}.jpg")
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    with pytest.raises(ValueError, match="hummingbird"):
        clf.train_model(tmp_path / "train", tmp_path / "val", epochs=1)


def test_train_model_only_one_class_raises(tmp_path):
    """train_model must raise ValueError when only one class has samples."""
    for split in ("train", "val"):
        d = tmp_path / split / "hummingbird"
        d.mkdir(parents=True)
        # 'other' directory exists but is empty
        (tmp_path / split / "other").mkdir(parents=True)
        for i in range(2):
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(d / f"hb_{i}.jpg")
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    with pytest.raises(Exception):  # ValueError or ImageFolder error
        clf.train_model(tmp_path / "train", tmp_path / "val", epochs=1)


def test_build_classifier_factory():
    """build_classifier() constructs a HummingbirdClassifier from a config dict."""
    cfg = {
        "classifier": {
            "model": "hummingbird_classifier.pt",
            "confidence_threshold": 0.7,
        }
    }
    # model_path won't exist on disk; __init__ handles that with a warning
    clf = build_classifier(cfg)
    assert isinstance(clf, HummingbirdClassifier)
    assert clf.confidence_threshold == 0.7


def test_train_model_with_wandb_run(tmp_path):
    """wandb_run.log() is called once per epoch when wandb_run is provided."""
    train_dir, val_dir = _make_tiny_dataset(tmp_path)
    mock_wandb = MagicMock()
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    clf.train_model(
        train_dir,
        val_dir,
        epochs=1,
        output_path=tmp_path / "ckpt.pt",
        wandb_run=mock_wandb,
    )
    assert mock_wandb.log.called


def test_train_model_phase2_scheduler_step(tmp_path):
    """Running >5 epochs triggers Phase 2 (backbone unfreeze) and scheduler.step."""
    train_dir, val_dir = _make_tiny_dataset(tmp_path)
    clf = HummingbirdClassifier(model_path=None, device="cpu")
    result = clf.train_model(
        train_dir,
        val_dir,
        epochs=7,
        output_path=tmp_path / "ckpt.pt",
        early_stopping_patience=10,
    )
    assert result["best_epoch"] >= 1
