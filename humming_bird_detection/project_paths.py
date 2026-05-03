"""Repo-relative path constants for the hummingbird detection package.

All paths are resolved relative to the repository root, which is determined
as follows (in order of priority):

1. The ``HBD_ROOT`` environment variable, if set.  Use this when the repo is
   checked out at a non-standard location or when running on a cluster where
   data lives on a scratch partition::

       export HBD_ROOT=/scratch/myuser/Humming_Bird_Detection

2. The directory two levels above this file (i.e. the repo root), which is
   correct for any standard checkout.

Import style (unchanged from previous API):

    from humming_bird_detection.project_paths import project_paths
    config_path = project_paths.data.interim / "pipeline.yaml"
    model_dir   = project_paths.models
"""

import os
from pathlib import Path
from types import SimpleNamespace

_root = Path(os.environ.get("HBD_ROOT", Path(__file__).parents[1])).resolve()

project_paths = SimpleNamespace(
    root=_root,
    data=SimpleNamespace(
        raw=_root / "data" / "raw",
        external=_root / "data" / "external",
        interim=_root / "data" / "interim",
        processed=_root / "data" / "processed",
    ),
    models=_root / "models",
    notebooks=_root / "notebooks",
    references=_root / "references",
    reports=_root / "reports",
    figures=_root / "reports" / "figures",
)
