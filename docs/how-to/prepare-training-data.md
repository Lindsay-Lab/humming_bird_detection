# Prepare training data

Training data consists of bird crops extracted from camera-trap images. This guide covers the two supported sources: **Roboflow** (annotated dataset) and **iNaturalist** (web download).

## From a Roboflow export

### 1. Export from Roboflow

In the Roboflow UI, export your dataset in **YOLOv8** format and place the zip at `data/raw/roboflow/`.

### 2. Run the preparation script

```bash
python scripts/prepare_training_data.py \
    --source roboflow \
    --roboflow-dir data/raw/roboflow \
    --output-dir   data/interim/train_data \
    --val-split    0.2 \
    --seed         42
```

The script:
- Decodes YOLO-format bounding boxes
- Crops and pads each annotation (`pad_fraction` from `pipeline.yaml`)
- Copies crops to `train/hummingbird/`, `train/other/`, `val/hummingbird/`, `val/other/`

---

## From iNaturalist

### 1. Download images

```bash
python humming_bird_detection/data/load_inaturalist.py \
    --taxon_name Trochilidae \
    --output data/raw/Trochilidae \
    --per_page 100
```

This downloads up to 100 research-grade hummingbird photos per page and continues until all pages are exhausted.

### 2. Generate YOLO labels

```bash
bash data/raw/Trochilidae/make_labels.sh
```

### 3. Run the preparation script

```bash
python scripts/prepare_training_data.py \
    --source inaturalist \
    --image-dir  data/raw/Trochilidae/images \
    --label-dir  data/raw/Trochilidae/labels \
    --output-dir data/interim/train_data \
    --val-split  0.2 \
    --seed       42
```

---

## Verify the output

```
data/interim/train_data/
    train/
        hummingbird/   N images
        other/         M images
    val/
        hummingbird/
        other/
```

Check class balance before training — a severely imbalanced dataset will be compensated by the automatic `pos_weight` in the loss function, but very extreme ratios (> 50:1) may require collecting more data.
