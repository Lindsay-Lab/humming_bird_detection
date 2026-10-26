# Training Scripts

Scripts for preparing training data and fine-tuning the EfficientNetV2-S hummingbird classifier.  Training is designed to run on an HPC cluster with Slurm and Singularity — the SLURM scripts reference NYU Greene–specific paths and must be adapted for other clusters.

---

## Scripts overview

| Script | Purpose |
|--------|---------|
| `prepare_training_data.py` | Build `train/` and `val/` ImageFolder directories from raw data sources. |
| `train_classifier.py` | Fine-tune the classifier (standalone or launched by SLURM). |
| `train_classifier.slurm` | Submit a single training run to Slurm. |
| `train_classifier_seed_sweep.slurm` | Submit a seed sweep (10 seeds, array job) to Slurm. |

---

## Prerequisites

### 1. Annotated training data

The classifier requires labeled images of hummingbirds and non-hummingbird birds visiting the feeder.  Two data-preparation modes are supported:

- **Mode A — Roboflow export (recommended).**  Export your annotated dataset from Roboflow in YOLOv8/v11 format.  The export produces a directory tree with `train/`, `valid/`, and `test/` splits, each containing `images/` and `labels/` subdirectories.

- **Mode B — Pre-collected directories.**  Directories of hummingbird images (e.g. downloaded from iNaturalist) and directories of other bird images.  Images are copied as-is without cropping.

### 2. WandB API key

Training logs metrics to Weights & Biases.  Store your key:

```bash
# One-time setup
echo "your-api-key-here" > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

### 3. Slurm account

The SLURM scripts use `--account=torch_pr_628_general`.  If you are on a different cluster or account, edit the `#SBATCH --account=…` line in both `.slurm` files.

### 4. Singularity container (Greene-specific)

Training runs inside a Singularity container with CUDA 12.2.2 and a Conda overlay.  The overlay path (`/scratch/rg5022/images/rva.ext3`) is hardcoded in the SLURM scripts — adapt to your cluster's image setup.

---

## Step 1 — Prepare training data

### Mode A — from Roboflow YOLOv8/v11 export

```bash
# 1. Download the dataset from Roboflow as YOLOv11
# 2. Unzip into data/raw/, e.g.:
#    data/raw/label-birdfeeder-camera-observations-3/

# 3. Crop birds from YOLO annotations and write train/val splits
python scripts/prepare_training_data.py \
    --roboflow-dir data/raw/label-birdfeeder-camera-observations-3 \
    --out-root data/interim/train_data
```

This pools images from all Roboflow splits, crops each annotated bounding box (+10 % padding) to 224×224, assigns crops to `hummingbird/` or `other/` by class name, and performs a fresh 80/20 stratified split.  Annotations whose class name contains "nobird" / "no_bird" are skipped.

Output layout:

```
data/interim/train_data/
├── train/
│   ├── hummingbird/   (1823 crops)
│   └── other/         (2165 crops)
└── val/
    ├── hummingbird/   (456 crops)
    └── other/         (542 crops)
```

### Mode B — from pre-collected image directories

```bash
python scripts/prepare_training_data.py \
    --hummingbird-src data/raw/Trochilidae/images \
    --other-src data/raw/other_birds/Tyrannidae data/raw/other_birds/Corvidae \
    --out-root data/interim/train_data
```

Images are copied as-is (no cropping).  Multiple `--other-src` directories are merged.  One image = one training sample.

### Customizing the split

```bash
python scripts/prepare_training_data.py \
    --roboflow-dir data/raw/label-birdfeeder-camera-observations-3 \
    --out-root data/interim/train_data \
    --val-split 0.15 \
    --seed 123
```

---

## Step 2 — Submit a training job

### Single training run

```bash
# From the project root:
export WANDB_API_KEY=$(cat ~/.wandb_api_key)
sbatch scripts/train_classifier.slurm
```

This submits one job requesting 1 GPU, 8 CPUs, 64 GB RAM, and a 6-hour walltime.  The script runs:

```bash
python scripts/train_classifier.py \
    --train-dir  data/interim/train_data/train \
    --val-dir    data/interim/train_data/val \
    --epochs     50 \
    --batch-size 128 \
    --lr         1.2e-3 \
    --weight-decay 1e-4 \
    --early-stopping-patience 5
```

### Seed sweep (10 seeds)

```bash
export WANDB_API_KEY=$(cat ~/.wandb_api_key)
sbatch scripts/train_classifier_seed_sweep.slurm
```

Submits an array job with `--array=0-9`, one task per seed (0–9).  Each task runs the same hyperparameters with a different random seed.

### Monitoring

```bash
# Job status
squeue -u $USER

# Job output
tail -f /scratch/rg5022/Humming_Bird_Detection/logs/train_classifier_<jobid>.log

# WandB dashboard
# The run URL is printed at the top of the log output.
```

### Customizing hyperparameters

Edit the SLURM script directly:

```bash
python scripts/train_classifier.py \
    --train-dir  data/interim/train_data/train \
    --val-dir    data/interim/train_data/val \
    --epochs     100 \              # more epochs
    --batch-size 64 \               # smaller batch
    --lr         3e-4 \             # lower learning rate
    --weight-decay 1e-4 \
    --early-stopping-patience 10    # more patience
```

Or run locally (no Slurm, no WandB):

```bash
python scripts/train_classifier.py \
    --train-dir data/interim/train_data/train \
    --val-dir data/interim/train_data/val \
    --epochs 10 \
    --no-wandb
```

---

## Step 3 — View results

### Output files

For each run, the script writes to `models/`:

```
models/
├── hummingbird_classifier_YYYYMMDD_HHMMSS.pt            # best checkpoint
├── hummingbird_classifier_YYYYMMDD_HHMMSS_metadata.json  # provenance & hyperparams
├── hummingbird_classifier_YYYYMMDD_HHMMSS_wandb_model_metrics.csv   # per-epoch metrics
├── hummingbird_classifier_YYYYMMDD_HHMMSS_wandb_system_metrics.csv  # GPU/CPU/memory
└── hummingbird_classifier_YYYYMMDD_HHMMSS_wandb_output.log          # captured stdout
```

### Metadata file

The `_metadata.json` file records:

```json
{
  "run_id": "20260430_193523",
  "timestamp_utc": "2026-04-30T19:35:23.123456+00:00",
  "slurm_job_id": "12345678",
  "git_commit": "3ad7016",
  "wandb": {
    "run_id": "abc123",
    "run_url": "https://wandb.ai/…",
    "project": "hummingbird-detection",
    "metadata": { … },
    "summary": { "best_val_loss": 0.042, "best_epoch": 23 }
  },
  "hyperparameters": {
    "epochs": 50,
    "batch_size": 128,
    "lr": 0.0012,
    "seed": 42
  },
  "results": {
    "best_val_loss": 0.042,
    "best_epoch": 23
  }
}
```

### Generating a training report

```bash
python docs/scripts/generate_training_report.py \
    models/hummingbird_classifier_YYYYMMDD_HHMMSS.pt \
    --output-dir docs/training-reports
```

### Using the trained checkpoint

Copy or symlink the best checkpoint to the expected path:

```bash
cp models/hummingbird_classifier_YYYYMMDD_HHMMSS.pt models/hummingbird_classifier.pt
```

---

## Adapting to a different cluster

1. **SLURM account** — change `#SBATCH --account=…` to your account.
2. **Output paths** — change `#SBATCH --output` and `#SBATCH --error` paths.
3. **Singularity container** — replace the `.sif` path and `--overlay` path.
4. **Log directory** — update the `mkdir -p` line.
5. **Project root** — update the `cd /scratch/rg5022/Humming_Bird_Detection` line.
6. **Conda environment** — replace `conda activate hbd` with your environment name.
