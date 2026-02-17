#paths.py
from __future__ import annotations
from pathlib import Path

THIS_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../humming_bird_detection/
PROJ_ROOT = THIS_PACKAGE_ROOT.parent                        # repo root

LOGS_DIR = PROJ_ROOT / "logs"

DATA_DIR = PROJ_ROOT / "data"
#RAW_DATA_DIR = DATA_DIR / "raw"  
RAW_DATA_DIR = THIS_PACKAGE_ROOT / "Label-Birdfeeder-Camera-Observations-3"         
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"