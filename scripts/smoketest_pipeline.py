"""Smoketest the full pipeline on data/raw/13_WSCT0033.JPG."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from humming_bird_detection.config import load_config
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.models.classifier import build_classifier
from humming_bird_detection.workflow.pipeline import process_directory
from humming_bird_detection.workflow.analysis import generate_report

cfg = load_config()
print("Loading models ...", flush=True)
detector = build_detector(cfg)
classifier = build_classifier(cfg)
print("Models loaded.", flush=True)

df = process_directory(
    "data/raw", "reports/smoketest_detections.csv", detector, classifier, cfg
)
print(f"DF shape: {df.shape}", flush=True)
print(f"DF columns: {list(df.columns)}", flush=True)
print(f"DF attrs: {df.attrs}", flush=True)

if not df.empty:
    print("\n--- Bird-level CSV (first 10 rows) ---")
    print(df.head(10).to_string(), flush=True)
    print("\n--- Analysis Report ---")
    print(generate_report(df, cfg), flush=True)
else:
    print("No birds detected in the image.", flush=True)

# Check per-photo CSV
import pandas as pd
photo_csv = "reports/smoketest_detections_per_photo.csv"
try:
    pdf = pd.read_csv(photo_csv)
    print(f"\n--- Per-photo CSV ({len(pdf)} rows) ---")
    print(pdf.head().to_string(), flush=True)
except FileNotFoundError:
    print(f"\nPer-photo CSV not found at {photo_csv}", flush=True)
