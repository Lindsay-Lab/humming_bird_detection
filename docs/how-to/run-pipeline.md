# Run the pipeline on a directory of images

## What you need

- Package installed (`pip install -e .`)
- YOLO checkpoint at `models/YOLO/yolo11n.pt`
- Classifier checkpoint at `models/hummingbird_classifier.pt`
- `data/interim/pipeline.yaml` present (copy from the example in the repo root)

## Before you start: runtime expectations

Detection cost is dominated by **SAHI tiled inference**, which only runs when
the fast full-image YOLO pass finds no birds. This means — counterintuitively —
**blank images are the slowest**: on a CPU-only machine a blank 8 MP image
takes **15–80 s** (the image is cut into ~45 tiles, each run through YOLO),
while an image with an easily visible bird takes ~1–3 s.

Rules of thumb:

| Setup | Blank image | Image with bird | 160-image folder (~70 % blanks) |
|-------|------------|-----------------|--------------------------------|
| CPU, default (tiling on) | 15–80 s | 1–3 s | **~45–60 min** |
| CPU, `--no-tiling` | ~1 s | ~1 s | **~3–5 min** |
| CUDA GPU | ~1–3 s | <1 s | a few minutes |

Because runs can be long, the pipeline:

- shows a **progress bar with ETA** (disable with `--no-progress`),
- **appends every record to a `<output>.partial.csv` sidecar immediately**, so
  an interrupted run never loses completed work,
- supports **`--resume`** to skip already-processed images and continue an
  interrupted run (the sidecar is deleted automatically on success).

### Fast mode vs. full mode

`--no-tiling` disables SAHI entirely (single full-image YOLO pass per photo).
It is ~10–20× faster on CPU but **may miss small or distant birds** that only
tiled inference can recover. Use fast mode for a quick first look or threshold
tuning, then re-run in full mode for the final dataset. To make fast mode the
default, set `detector.slice_size: null` in the config.

## Steps

### 1. Run via CLI (recommended)

```bash
# Full pipeline with report, JSON summary, and plots
hbd detect \
  --input-dir data/raw/station_A \
  --output-csv output/observations.csv \
  --report \
  --json-summary output/summary.json \
  --plots-dir output/plots/

# Quick first look (~10–20× faster on CPU, may miss small/far birds)
hbd detect --input-dir data/raw/station_A --output-csv output/obs.csv --no-tiling

# Continue an interrupted run from where it stopped
hbd detect --input-dir data/raw/station_A --output-csv output/observations.csv --resume

# Skip composition-based suspected-duplicate flagging
hbd detect --input-dir data/raw/station_A --output-csv output/obs.csv --no-suspected-duplicates

# Analyse an existing detection CSV (no re-detection — takes seconds)
hbd analyze \
  --input-csv output/observations.csv \
  --json-summary output/summary.json \
  --plots-dir output/plots/ \
  --per-photo-csv output/per_photo.csv
```

For long CPU runs, launch inside `tmux`/`screen` or with
`setsid nohup ... &` so the run survives a closed terminal; `--resume` covers
you if it is killed anyway.

### 2. Run from Python

```python
from humming_bird_detection.config import load_config
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.models.classifier import build_classifier
from humming_bird_detection.workflow.pipeline import process_directory

cfg        = load_config()                  # reads data/interim/pipeline.yaml
detector   = build_detector(cfg)
classifier = build_classifier(cfg)

df = process_directory(
    input_dir="data/raw/station_A",
    output_csv="data/processed/station_A_observations.csv",
    detector=detector,
    classifier=classifier,
    cfg=cfg,
    resume=False,     # True: skip images already in the .partial.csv sidecar
    progress=True,    # tqdm progress bar (falls back to periodic log lines)
)
```

To use a different config file: `cfg = load_config("path/to/my_config.yaml")`.

`process_directory` processes all `.jpg`/`.JPG`/`.jpeg`/`.JPEG` files in
`input_dir`, writes two CSVs, and returns the bird-level DataFrame. The
returned DataFrame carries `df.attrs["total_images"]` and
`df.attrs["blank_images"]` for downstream analysis.

For a single image:

```python
from humming_bird_detection.workflow.pipeline import process_image

records = process_image("data/raw/station_A/IMG_0042.JPG", detector, classifier, cfg)
# records is a list of dicts (one per detected bird, empty if no birds)
```

### 3. Analyse the results

```python
from humming_bird_detection.workflow.analysis import generate_report, export_summary_json, generate_plots

print(generate_report(df, cfg))                  # terminal report
export_summary_json(df, "analysis/summary.json", cfg)
generate_plots(df, "analysis/plots/", cfg)       # 4 PNGs
```

The report includes camera efficiency (blanks vs birds), per-camera
breakdowns, hourly/daily activity distributions, confidence statistics, and
duplicate analysis.

### 4. Export per-photo aggregation

```python
from humming_bird_detection.workflow.postprocessing import aggregate_to_photos

photos_df = aggregate_to_photos(df)
photos_df.to_csv("data/processed/station_A_per_photo.csv", index=False)
```

## Understanding the outputs

### Bird-level CSV (one row per detected bird)

| Column | Meaning |
|--------|---------|
| `filename` | Source image file name |
| `date`, `time` | OCR-extracted timestamp, normalised to `DD MON YYYY` / `HH:MM AM/PM` (minute precision) |
| `temperature` | OCR-extracted temperature in °F (`None` if unreadable) |
| `camera` | OCR-extracted camera label, e.g. `CAMERA 3` |
| `bird_index` | 0-based index of the bird within its photo |
| `bbox_x1/y1/x2/y2` | Detection bounding box in original-image pixel coordinates — use these to re-crop or review detections |
| `detection_confidence` | YOLO detection score (0–1) |
| `classification_confidence` | **Sigmoid probability of the *hummingbird* class.** For birds classified as "other" this is *low* (e.g. 0.0001 = very certain it is NOT a hummingbird). Do not average this column across mixed classes. |
| `is_hummingbird` | `True` when `classification_confidence ≥ classifier.confidence_threshold` |
| `is_duplicate` | Same motion event as a previous photo (within `deduplication.window_seconds`) |
| `is_suspected_duplicate` | Composition-based duplicate within the same minute (ceil(N/3) rule) |

!!! note "Confidence vs. certainty"
    `classification_confidence` is P(hummingbird), so it is *inversely* coded
    for "other" birds. The report and JSON therefore also provide
    **classifier certainty** = confidence in the *predicted* class
    (`p` for hummingbirds, `1 − p` for others; 1.0 = fully certain).
    A well-behaved classifier shows certainty near 1.0 for both classes —
    the raw column mean near 0.5 is expected and *not* a sign of
    chance-level performance.

### Per-photo CSV (`_per_photo.csv`, one row per photo)

Aggregated counts (`num_total_birds`, `num_hummingbirds`, `num_other`,
`has_hummingbird`), confidence stats (mean/max), and propagated duplicate
flags. **Blank photos (no detections) do not appear in either CSV** — total
and blank counts are only available in the run report and JSON summary
produced by `hbd detect` (see caveat below).

### `.partial.csv` sidecar

While a run is in progress you will see `<output>.partial.csv` next to your
output path. It is the raw incremental record stream (blank photos appear as
sentinel rows with `bird_index = -1`) and is deleted automatically when the
run completes. Leave it in place to be able to `--resume`.

## OCR metadata caveats

Metadata (`date`, `time`, `temperature`, `camera`) is read from the camera's
bottom info strip with EasyOCR, which is imperfect:

- The `°F` glyph is frequently misread as `%`, `P`, or a quote; the parser
  accepts these variants and searches only the text *before* the `CAMERA`
  token so timestamp digits are never mistaken for a temperature.
- Time spacing is inconsistent (`02 : 28PM`, `10 :55AM`); times are normalised
  to `HH:MM AM/PM` so downstream parsing (deduplication, hourly/daily
  activity) works reliably.
- Fields that still cannot be parsed are stored as `None` and logged as
  warnings — check stderr for `Could not parse field` messages. Rows with
  missing timestamps are conservatively treated as non-duplicates and excluded
  from time-based charts.
- Timestamps have **one-minute resolution**; the `deduplication.window_seconds`
  default (18 s) therefore effectively flags only photos with the *same*
  minute stamp.

## Caveat: `hbd analyze` and blank counts

Blank-image totals live in `df.attrs`, which a CSV cannot carry. When you run
`hbd analyze --input-csv ...` on an existing CSV, the report's "Total images"
equals the number of photos *with birds*, and blanks show as 0. For complete
efficiency statistics use the report/JSON produced by `hbd detect` at run
time.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Run takes ~1 min per image | CPU + SAHI fallback on blank images | Use `--no-tiling` for a first pass, or run on a CUDA machine |
| Run was killed / terminal closed | — | Re-run the same command with `--resume`; completed images are skipped |
| No detections | `confidence_threshold` too high | Lower `detector.confidence_threshold` in config |
| Many false positives | Threshold too low | Raise `detector.confidence_threshold` |
| Temperature/time fields `None` | OCR could not read the strip | Check `margin_px` matches your camera's strip height; inspect the `Could not parse` warnings for the raw OCR text |
| Hourly/Daily sections empty | Unparseable timestamps | Inspect `date`/`time` columns; ensure they follow `DD MON YYYY` / `HH:MM AM/PM` |
| "Classification conf. ~0.5" looks like chance | Averaging raw P(hummingbird) across classes | Read "Classifier certainty" instead — see note above |
| `FileNotFoundError` on config | Missing `pipeline.yaml` | Copy `data/interim/pipeline.yaml` from the repo |
