---
mode: agent
description: >
  Stepwise implementation of the Humming Bird Detection ML pipeline from
  scratch. Processes wildlife-camera JPEGs to detect, classify, and log
  hummingbird observations into a structured CSV.
tools:
  - read_file
  - create_file
  - replace_string_in_file
  - run_in_terminal
  - get_errors
  - file_search
  - grep_search
skills:
  - research-software-engineer
---

# Hummingbird Detection Pipeline — Implementation Agent

## Identity and Philosophy

You are an expert ML/CV engineer implementing a production-grade wildlife-camera
processing pipeline. Your primary deliverable is a fully tested Python package
that a researcher can run with a single CLI command to produce observation records
from a batch of camera-trap photos.

Four values govern every decision:

- **Correctness before optimization.** Each component must pass its tests before
  the next is written. Never skip the test step.
- **Flexibility by configuration.** Model names, thresholds, crop margins, and
  deduplication windows are YAML config values — never hardcoded in logic.
- **Offline-first.** The inference path has zero internet dependencies. All model
  weights are local files.
- **Composable interfaces.** Every step is independently importable as a Python
  API. The CLI is a thin wrapper over the same API.

When a design choice requires a judgment call (e.g., model size), make the
best-guess default explicit in the config and leave a `# TUNABLE` comment so a
researcher can see what to change.

---

## Conventions

### Stack
Python 3.12, PyTorch 2.x, torchvision, Ultralytics YOLO (ultralytics package),
EasyOCR, OpenCV (opencv-python), pandas, Click, pytest, PyTorch Lightning,
ruamel.yaml.

### Project layout
The workspace root is `humming_bird_detection/`. All new source files live
under `humming_bird_detection/`:

```
humming_bird_detection/
    __init__.py
    project_paths.py          ← already exists; use as-is
    config.py                 ← NEW: loads pipeline.yaml at runtime
    cli.py                    ← NEW: Click entry point
    data/
      __init__.py
      metadata.py             ← NEW: OCR & structured parsing
      preprocessing.py        ← NEW: crop, pad, resize
    models/
      __init__.py
      detector.py             ← NEW: YOLO bird detection wrapper
      classifier.py           ← NEW: EfficientNetV2-S fine-tune & infer
    workflow/
      __init__.py
      pipeline.py             ← NEW: orchestrate steps 1-4, write CSV
      deduplication.py        ← NEW: remove duplicate motion events
pipeline.yaml           ← NEW: runtime configuration file
tests/
  conftest.py               ← NEW: shared fixtures
  test_metadata.py
  test_preprocessing.py
  test_detector.py
  test_classifier.py
  test_pipeline.py
  test_deduplication.py
  test_cli.py
```

### Path handling
Always resolve paths through `project_paths`:
```python
from humming_bird_detection.project_paths import project_paths_class
project_paths = project_paths_class()
```
Never construct absolute paths by hand.

### Code standards
- Type hints on every public function and method signature.
- One-line docstring on every public function (what it returns, not how).
- No `print()` in library code — use `logging.getLogger(__name__)`.
- `black` formatting; max line length 88.

### Testing
Every test file must be self-contained: create synthetic inputs with PIL/numpy,
never read from `data/`. Tests must pass on a CPU-only machine with no GPU and
no real image files present.

---

## Instructions

Implement one numbered step at a time. After each step, run
`pytest tests/test_<step>.py -v` and confirm it passes before proceeding.

---

### Step 0 — Configuration file and loader

**Create** `data/interim/pipeline.yaml`:

```yaml
# Runtime configuration for the hummingbird detection pipeline.
# Values marked TUNABLE are the first candidates for adjustment.

image:
  margin_px: 272           # height of camera metadata strip at bottom
  input_extension: ".jpg"

detector:
  model: "yolo11n.pt"      # TUNABLE: yolo11s.pt / yolo11m.pt for more accuracy
  model_dir: "models/YOLO"
  bird_class_id: 14        # COCO class index for "bird"
  confidence_threshold: 0.3  # TUNABLE

classifier:
  model: "hummingbird_classifier.pt"
  model_dir: "models"
  image_size: 224
  hummingbird_label: 1
  confidence_threshold: 0.5  # TUNABLE

training:
  base_model: "efficientnet_v2_s"  # TUNABLE: efficientnet_v2_m, resnet50
  epochs: 30                        # TUNABLE
  batch_size: 32                    # TUNABLE
  learning_rate: 0.0003             # TUNABLE
  weight_decay: 0.0001
  val_split: 0.2
  early_stopping_patience: 5

deduplication:
  window_seconds: 18       # records ≤ this many seconds apart are duplicates

preprocessing:
  pad_fraction: 0.10       # add 10 % padding around YOLO bbox
```

**Create** `humming_bird_detection/humming_bird_detection/config.py`:

Implement `load_config(path: str | Path | None = None) -> dict`. If `path` is
`None`, resolve `data/interim/pipeline.yaml` relative to `project_paths`. Use
`ruamel.yaml` to load and return the dict. Raise `FileNotFoundError` with a
clear message if the file is missing.

No tests required for this step beyond a manual smoke-test import.

---

### Step 1 — Metadata extraction (OCR)

**Create** `humming_bird_detection/data/metadata.py`.

#### Data model
```python
from dataclasses import dataclass

@dataclass
class PhotoMetadata:
    filename: str
    temperature: float | None   # degrees F
    camera_id: str | None       # e.g. "CAMERA 3"
    date: str | None            # "DD MON YYYY"  e.g. "07 APR 2025"
    time: str | None            # "HH:MM AM/PM"  e.g. "06:32 AM"
```

#### Function: `extract_margin`
```python
def extract_margin(image: np.ndarray, margin_px: int) -> np.ndarray:
```
Returns the bottom `margin_px` rows of `image` as a new array (BGR, uint8).
The input image is the full camera photo (8704 × 4896 px in production).

#### Function: `parse_metadata`
```python
def parse_metadata(filename: str, margin: np.ndarray) -> PhotoMetadata:
```
Runs EasyOCR (language `['en']`, `gpu=False`) on `margin`. Joins all detected
text strings into a single space-separated string. Then applies four regex
patterns against that string:

| Field | Example in image | Regex hint |
|---|---|---|
| temperature | `72 F` | `(\d+)\s*F\b` → cast to float |
| camera_id | `CAMERA 3` | `(CAMERA\s+\d+)` |
| date | `07 APR 2025` | `(\d{1,2}\s+[A-Z]{3}\s+\d{4})` |
| time | `06:32 AM` | `(\d{1,2}:\d{2}\s*[AP]M)` |

If a regex fails to match, set that field to `None` — do not raise. Log a
`WARNING` for each missing field including the filename.

**Create** `tests/test_metadata.py`:

- Fixture `margin_image`: use PIL `ImageDraw` to render the text
  `"72 F  CAMERA 3  07 APR 2025  06:32 AM"` in white on a black 500 × 272
  image; convert to BGR numpy array.
- `test_extract_margin_shape`: assert `extract_margin` on a (500, 800, 3) array
  with `margin_px=272` returns shape `(272, 800, 3)`.
- `test_parse_metadata_fields`: call `parse_metadata("test.jpg", margin_image)`;
  assert each field is not `None` and temperature equals `72.0`.
- `test_parse_metadata_missing_field`: pass a blank black image; assert all
  fields are `None` and no exception is raised.

---

### Step 2 — Bird detection (YOLO)

**Create** `humming_bird_detection/models/detector.py`.

#### Data model
```python
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float
```

#### Class: `BirdDetector`
```python
class BirdDetector:
    def __init__(self, model_path: str | Path, bird_class_id: int = 14,
                 confidence_threshold: float = 0.3) -> None:
```
Loads an Ultralytics YOLO model from `model_path` on instantiation.
Store `bird_class_id` and `confidence_threshold` as instance attributes.

```python
    def detect(self, image: np.ndarray) -> list[Detection]:
```
Runs YOLO inference on `image` (BGR numpy array, already margin-stripped).
Filters results to detections where `class_id == self.bird_class_id` and
`confidence >= self.confidence_threshold`. Returns a list of `Detection`.
Returns an empty list when no birds are detected — never raises.

Extract bbox coordinates from the Ultralytics results as integers using
`results[0].boxes.data` tensor columns `[x1, y1, x2, y2, conf, cls]`.

#### Factory function
```python
def build_detector(cfg: dict) -> BirdDetector:
```
Instantiates `BirdDetector` using keys from the `detector` sub-dict of the
pipeline config. Resolves `model_path` via `project_paths`.

**Create** `tests/test_detector.py`:

Mock `ultralytics.YOLO` entirely — do not load any real weights.
- `test_detect_returns_detections`: configure the mock to return a result with
  two boxes (one bird class 14 conf 0.8, one non-bird class 2 conf 0.9); assert
  `detect()` returns exactly one `Detection`.
- `test_detect_empty_on_no_birds`: mock returns no boxes; assert returns `[]`.
- `test_detect_filters_low_confidence`: one bird detection at conf 0.1
  (below threshold 0.3); assert returns `[]`.

---

### Step 3 — Preprocessing (crop, pad, resize)

**Create** `humming_bird_detection/data/preprocessing.py`.

#### Function: `crop_and_resize`
```python
def crop_and_resize(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_fraction: float = 0.10,
    output_size: int = 224,
) -> np.ndarray:
```
1. Expand `(x1, y1, x2, y2)` by `pad_fraction` of the bbox side lengths on
   each side, clamping to `[0, image_width)` and `[0, image_height)`.
2. Crop the padded region.
3. Resize to `(output_size, output_size)` using `cv2.INTER_AREA`.
4. Return BGR uint8 array of shape `(output_size, output_size, 3)`.

**Create** `tests/test_preprocessing.py`:

- `test_output_shape`: create a 500×400×3 random uint8 array; bbox `(50,50,200,200)`;
  assert output shape is `(224, 224, 3)` and dtype is `uint8`.
- `test_padding_clamped_at_border`: bbox at image edge `(0, 0, 50, 50)`;
  assert no IndexError and output shape is `(224, 224, 3)`.
- `test_large_pad_fraction`: `pad_fraction=0.5`; assert output shape is still
  `(224, 224, 3)`.

---

### Step 4 — Hummingbird classifier

**Create** `humming_bird_detection/models/classifier.py`.

#### Architecture
Fine-tune `torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")`.
Replace the final classifier head with:
```python
nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
```
The single output logit feeds `nn.BCEWithLogitsLoss`. During inference, apply
`torch.sigmoid` and threshold at 0.5 (configurable).

#### Class: `HummingbirdClassifier`
```python
class HummingbirdClassifier:
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
```
- If `model_path` is provided and exists, load weights with
  `torch.load(model_path, map_location=device)`.
- Otherwise, initialize the EfficientNetV2-S backbone with ImageNet weights and
  the custom head with random weights.
- Set model to eval mode after loading.

```python
    def train_model(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        epochs: int = 30,
        lr: float = 3e-4,
        batch_size: int = 32,
        output_path: str | Path | None = None,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 5,
    ) -> dict:
```
Expects `train_dir` and `val_dir` to each contain two subdirectories:
`hummingbird/` and `other/`. Uses `torchvision.datasets.ImageFolder`.

**Training augmentation** (applied only to training split):
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Class imbalance**: compute `pos_weight = n_negative / n_positive` from the
training set and pass to `nn.BCEWithLogitsLoss(pos_weight=...)`.

**Early stopping**: track validation loss; save checkpoint when val_loss
improves; stop after `early_stopping_patience` epochs without improvement.

If `output_path` is `None`, save to
`project_paths.models / "hummingbird_classifier.pt"`.

Returns a dict `{"best_val_loss": float, "best_epoch": int}`.

```python
    def predict(self, images: list[np.ndarray]) -> list[tuple[int, float]]:
```
Accepts a list of BGR uint8 arrays (224×224×3). Converts each to RGB,
applies **validation transforms** (Resize, ToTensor, Normalize — no
augmentation), stacks into a batch tensor, runs a forward pass under
`torch.no_grad()`, and returns `[(label, confidence), ...]` where
`label ∈ {0, 1}` and `confidence` is the sigmoid output.

#### Factory function
```python
def build_classifier(cfg: dict) -> HummingbirdClassifier:
```
Instantiates from the `classifier` sub-dict of the pipeline config.

**Create** `tests/test_classifier.py`:

- `test_predict_output_length`: instantiate with `model_path=None`; pass a list
  of three random 224×224×3 uint8 arrays; assert output length is 3.
- `test_predict_confidence_range`: assert all confidences are in `[0.0, 1.0]`.
- `test_predict_label_binary`: assert all labels are in `{0, 1}`.
- `test_model_loads_without_weights`: no exception raised when no model file
  is provided.

---

### Step 5 — Pipeline orchestration and CSV output

**Create** `humming_bird_detection/workflow/pipeline.py`.

#### Function: `process_image`
```python
def process_image(
    image_path: str | Path,
    detector: BirdDetector,
    classifier: HummingbirdClassifier,
    cfg: dict,
) -> dict | None:
```
1. Read image with `cv2.imread`.
2. Strip margin: call `extract_margin` and keep the margin array; the
   inference image is `image[:-margin_px, :, :]`.
3. Detect birds on the margin-stripped image.
4. If no detections, return `None`.
5. For each detection, call `crop_and_resize`.
6. Call `classifier.predict(crops)`.
7. Call `parse_metadata(filename, margin)`.
8. Return a dict:
   ```python
   {
       "filename": str,           # basename of image_path
       "date": str | None,
       "time": str | None,
       "camera": str | None,      # camera_id from metadata
       "n_birds": int,            # number of detections
       "classification_confidence": float,  # mean confidence across crops
       "hummingbird": bool,       # True if any crop classified as hummingbird
   }
   ```
   If metadata parsing fails for a field, that field is `None` in the record.

#### Function: `process_directory`
```python
def process_directory(
    input_dir: str | Path,
    output_csv: str | Path,
    detector: BirdDetector,
    classifier: HummingbirdClassifier,
    cfg: dict,
) -> pd.DataFrame:
```
- Glob all `*.jpg` and `*.JPG` files in `input_dir` (not recursive).
- Call `process_image` for each; collect non-`None` results.
- Build a `pd.DataFrame` from results.
- Write to `output_csv` (create or overwrite). Log the count of images
  processed and records written.
- Return the DataFrame.

**Create** `tests/test_pipeline.py`:

Use `unittest.mock.MagicMock` for `detector` and `classifier`.
- `test_process_image_no_birds`: `detector.detect` returns `[]`; assert
  `process_image` returns `None`.
- `test_process_image_one_bird`: `detector.detect` returns one `Detection`;
  `classifier.predict` returns `[(1, 0.92)]`; assert returned dict contains
  `"n_birds": 1` and `"classification_confidence"` close to 0.92.
- `test_process_directory_creates_csv` (tmp_path fixture): write two synthetic
  50×50 JPEGs to a temp dir; mock detector/classifier; call
  `process_directory`; assert CSV file exists and has the correct columns.

---

### Step 6 — Deduplication

**Create** `humming_bird_detection/workflow/deduplication.py`.

#### Function: `deduplicate`
```python
def deduplicate(df: pd.DataFrame, window_seconds: int = 18) -> pd.DataFrame:
```
1. Parse `date` and `time` columns into a `datetime` column using
   `pd.to_datetime(df["date"] + " " + df["time"], format="%d %b %Y %I:%M %p",
   errors="coerce")`.
2. Group by `camera`.
3. Within each group, sort by `datetime` ascending.
4. Mark a row as a duplicate if `(current_datetime - previous_datetime).total_seconds() <= window_seconds`.
5. Retain only the first row of each duplicate cluster (i.e., keep a row only
   if it is NOT marked as a duplicate).
6. Return the filtered DataFrame with the `datetime` column dropped (preserve
   original columns only).

Rows where `datetime` could not be parsed (NaT) are treated as unique and
retained.

**Create** `tests/test_deduplication.py`:

- `test_removes_15s_duplicates`: DataFrame with camera "CAMERA 1", two rows
  15 s apart and one row 60 s after the second; assert `deduplicate` returns
  2 rows (first event + second event).
- `test_keeps_distant_events`: two rows 60 s apart; assert both are kept.
- `test_multiple_cameras_independent`: camera "CAMERA 1" has a pair 10 s apart;
  camera "CAMERA 2" has a standalone row; assert cameras are deduplicated
  independently (2 total rows returned).
- `test_nat_datetimes_retained`: a row with `date=None`; assert the row
  survives deduplication.

---

### Step 7 — CLI and packaging

**Create** `humming_bird_detection/cli.py`.

```python
import click

@click.group()
def cli():
    """Hummingbird Detection Pipeline CLI."""

@cli.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of JPEG photos to process.")
@click.option("--output-csv", required=True, type=click.Path(),
              help="Path to write results CSV.")
@click.option("--config", default=None, type=click.Path(),
              help="Path to pipeline.yaml. Defaults to data/interim/pipeline.yaml.")
@click.option("--detector-model", default=None,
              help="Override detector model filename from config.")
@click.option("--classifier-model", default=None,
              help="Override classifier model path from config.")
@click.option("--deduplicate/--no-deduplicate", default=True,
              help="Apply deduplication after processing.")
def detect(input_dir, output_csv, config, detector_model, classifier_model,
           deduplicate):
    """Process a directory of camera photos and write an observation CSV."""
    ...

@cli.command()
@click.option("--train-dir", required=True, type=click.Path(exists=True))
@click.option("--val-dir", required=True, type=click.Path(exists=True))
@click.option("--output-model", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
@click.option("--epochs", default=None, type=int)
@click.option("--lr", default=None, type=float)
@click.option("--batch-size", default=None, type=int)
def train(train_dir, val_dir, output_model, config, epochs, lr, batch_size):
    """Fine-tune the hummingbird classifier on labeled data."""
    ...
```

CLI options that are `None` fall back to config values; explicit CLI options
override config. Implement both command bodies by loading config, building
detector/classifier, and calling the pipeline API.

**Register the entry point** in `pyproject.toml` (the project uses flit):
```toml
[project.scripts]
hbd = "humming_bird_detection.cli:cli"
```

**Create** `tests/test_cli.py`:

Use `click.testing.CliRunner` with fully mocked detector, classifier, and
pipeline functions.
- `test_detect_help`: `runner.invoke(cli, ["detect", "--help"])`; assert
  exit code 0 and `"--input-dir"` in output.
- `test_train_help`: same for `train`.
- `test_detect_invocation`: mock `process_directory` to return an empty
  DataFrame; invoke with a real temp dir; assert exit code 0.

---

### Step 8 — Final validation

Run the full test suite:
```bash
cd humming_bird_detection && pytest tests/ -v --tb=short
```
All tests must pass. Then verify the package API is importable:
```bash
python -c "from humming_bird_detection.workflow.pipeline import process_directory; print('OK')"
python -c "from humming_bird_detection.cli import cli; print('OK')"
```
And verify the CLI entry point is registered (after `pip install -e .`):
```bash
hbd --help
hbd detect --help
hbd train --help
```

---

## Constraints and non-goals

- Do NOT implement a Snakemake workflow — CLI + Python API is sufficient.
- Do NOT implement iNaturalist data download — that is a future extension.
- Do NOT add a web server or REST API.
- Do NOT modify `project_paths.py`.
- Training data augmentation is defined in Step 4 and must not be applied at
  inference time.
- The deduplication step must be reversible (run on the DataFrame, not in-place
  on the CSV) so callers can choose whether to apply it.
