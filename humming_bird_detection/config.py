"""Config loader for the hummingbird detection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from ruamel.yaml import YAML

from humming_bird_detection.project_paths import project_paths as _project_paths

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = _project_paths.data.interim / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load pipeline.yaml and return its contents as a plain dict."""
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Pipeline config not found at '{resolved}'. "
            "Create data/interim/pipeline.yaml or pass an explicit path."
        )
    yaml = YAML()
    with resolved.open() as fh:
        cfg = yaml.load(fh)
    logger.debug("Loaded config from %s", resolved)
    return dict(cfg)
