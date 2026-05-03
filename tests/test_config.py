"""Tests for humming_bird_detection.config."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from humming_bird_detection.config import load_config


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(dedent(text))


_MINIMAL_YAML = """\
    image:
      margin_px: 272
    detector:
      model: yolo11n.pt
      bird_class_id: 14
      confidence_threshold: 0.3
    classifier:
      model: hummingbird_classifier.pt
      image_size: 224
      confidence_threshold: 0.5
    preprocessing:
      pad_fraction: 0.10
"""


def test_load_config_returns_dict(tmp_path):
    cfg_file = tmp_path / "pipeline.yaml"
    _write_yaml(cfg_file, _MINIMAL_YAML)
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)


def test_load_config_correct_values(tmp_path):
    cfg_file = tmp_path / "pipeline.yaml"
    _write_yaml(cfg_file, _MINIMAL_YAML)
    cfg = load_config(cfg_file)
    assert cfg["image"]["margin_px"] == 272
    assert cfg["detector"]["confidence_threshold"] == pytest.approx(0.3)
    assert cfg["classifier"]["image_size"] == 224
    assert cfg["preprocessing"]["pad_fraction"] == pytest.approx(0.10)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline.yaml"):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_missing_section_raises(tmp_path):
    cfg_file = tmp_path / "pipeline.yaml"
    _write_yaml(
        cfg_file,
        """\
        image:
          margin_px: 272
        detector:
          model: yolo11n.pt
          bird_class_id: 14
          confidence_threshold: 0.3
        classifier:
          model: hummingbird_classifier.pt
          image_size: 224
          confidence_threshold: 0.5
        """,
    )  # missing 'preprocessing' section
    with pytest.raises(KeyError, match="preprocessing"):
        load_config(cfg_file)


def test_load_config_missing_key_raises(tmp_path):
    cfg_file = tmp_path / "pipeline.yaml"
    _write_yaml(
        cfg_file,
        """\
        image:
          margin_px: 272
        detector:
          model: yolo11n.pt
          confidence_threshold: 0.3
        classifier:
          model: hummingbird_classifier.pt
          image_size: 224
          confidence_threshold: 0.5
        preprocessing:
          pad_fraction: 0.10
        """,
    )  # missing detector.bird_class_id
    with pytest.raises(KeyError, match="bird_class_id"):
        load_config(cfg_file)


def test_load_config_extra_keys_allowed(tmp_path):
    """Extra keys in the YAML must not raise — future-proofing."""
    cfg_file = tmp_path / "pipeline.yaml"
    _write_yaml(
        cfg_file,
        """\
        image:
          margin_px: 272
        detector:
          model: yolo11n.pt
          bird_class_id: 14
          confidence_threshold: 0.3
        classifier:
          model: hummingbird_classifier.pt
          image_size: 224
          confidence_threshold: 0.5
        preprocessing:
          pad_fraction: 0.10
        experimental:
          new_feature: true
        """,
    )
    cfg = load_config(cfg_file)
    assert cfg["experimental"]["new_feature"] is True
