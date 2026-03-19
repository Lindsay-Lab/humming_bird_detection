#paths.py
from __future__ import annotations
from pathlib import Path

THIS_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../humming_bird_detection/
PROJ_ROOT = THIS_PACKAGE_ROOT.parent                        # repo root

LOGS_DIR = PROJ_ROOT / "logs" / "yolo"

DATA_DIR = THIS_PACKAGE_ROOT / "data"
# Update this folder name after downloading from Roboflow in "YOLO" format.
# The YOLO annotation format is the same across YOLO26, YOLO11, YOLOv8, YOLOv5.
RAW_DATA_DIR = DATA_DIR / "raw" / "Label_Birdfeeder_Camera_Observations.v3i.yolo"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"