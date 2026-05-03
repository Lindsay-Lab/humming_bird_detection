# Getting started

This tutorial walks you from a fresh install to a processed observation CSV. By the end you will have run the full pipeline on a small set of camera images and understand what each step produces.

## Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda installed
- Camera-trap JPEG images with a metadata strip at the bottom (see [architecture](../explanation/architecture.md))
- A trained classifier checkpoint (`models/hummingbird_classifier.pt`), or use the ImageNet backbone for a first test run

---

## 1. Install the package

```bash
git clone https://github.com/rgutzen/Humming_Bird_Detection
cd Humming_Bird_Detection

conda env create -f environment.yml
conda activate humming_bird_detection
pip install -e .
```

Verify the install:

```bash
python -c "import humming_bird_detection; print('OK')"
```

---

## 2. Download the YOLO checkpoint

The pipeline uses YOLO11n pre-trained on COCO. Place the weight file at `models/YOLO/yolo11n.pt`:

```bash
mkdir -p models/YOLO
# download from https://github.com/ultralytics/assets/releases
wget -O models/YOLO/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

---

## 3. Check the configuration

Open `data/interim/pipeline.yaml`. The defaults work out of the box; the only value you might need to change is `image.margin_px` — set it to the pixel height of your camera's metadata strip.

```yaml
image:
  margin_px: 272   # adjust to your camera model
```

---

## 4. Run the pipeline

```python
from humming_bird_detection.config import load_config
from humming_bird_detection.models.detector import build_detector
from humming_bird_detection.models.classifier import build_classifier
from humming_bird_detection.workflow.pipeline import process_directory

cfg = load_config()
detector   = build_detector(cfg)
classifier = build_classifier(cfg)

df = process_directory(
    input_dir="path/to/your/images",
    output_csv="output/observations.csv",
    detector=detector,
    classifier=classifier,
    cfg=cfg,
)
print(df.head())
```

---

## 5. Inspect the output

The CSV has one row per detected bird:

| column | description |
|--------|-------------|
| `filename` | source image filename |
| `date`, `time` | from camera OCR |
| `camera` | camera ID from OCR |
| `hummingbird` | `1` = hummingbird, `0` = other bird |
| `classification_confidence` | model probability (0–1) |
| `detection_confidence` | YOLO confidence (0–1) |

---

## Next steps

- **Improve accuracy**: fine-tune the classifier on your own images — see [Fine-tune the classifier](../how-to/fine-tune-classifier.md).
- **Tune detection**: adjust `detector.confidence_threshold` and `detector.slice_size` — see [Configuration reference](../reference/config.md).
- **Understand the design**: see [Pipeline architecture](../explanation/architecture.md).
