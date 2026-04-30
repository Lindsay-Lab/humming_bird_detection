# Fine-tune the classifier

Fine-tuning adapts the hummingbird classifier to your specific camera setup, lighting conditions, and bird species mix. You need labelled image crops (hummingbird / other).

## What you need

- Training images organised as an `ImageFolder`-compatible directory tree (see [Prepare training data](prepare-training-data.md))
- GPU recommended; CPU works for small datasets

## Steps

### 1. Prepare your data directories

```
data/interim/train_data/
    train/
        hummingbird/   ← hummingbird crop images
        other/         ← non-hummingbird bird crops
    val/
        hummingbird/
        other/
```

### 2. Run training

```bash
python scripts/train_classifier.py \
    --train-dir data/interim/train_data/train \
    --val-dir   data/interim/train_data/val \
    --epochs    30 \
    --batch-size 32 \
    --seed      42
```

The script saves the best checkpoint to `models/hummingbird_classifier_<run_id>.pt` and a companion `<run_id>.json` with all hyperparameters and results.

### 3. Key options

| Flag | Default | Effect |
|------|---------|--------|
| `--epochs` | 30 | Total training epochs |
| `--batch-size` | 32 | Batch size (scale LR proportionally when changing) |
| `--lr` | 3e-4 | Head phase learning rate |
| `--early-stopping-patience` | 5 | Stop if val loss doesn't improve for N epochs |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-wandb` | off | Disable Weights & Biases logging |

### 4. Monitor training with W&B

```bash
export WANDB_API_KEY=<your_key>      # or: wandb login
python scripts/train_classifier.py --wandb-project my-project
```

### 5. Use the new checkpoint

Update `pipeline.yaml` to point to the new file:

```yaml
classifier:
  model: "hummingbird_classifier_20260430_143022.pt"
```

## Training phases

The script trains in two phases automatically:

1. **Epochs 1–5** — backbone frozen, only the classification head updated at full LR.
2. **Epoch 6+** — all weights unfrozen, LR reduced to `lr/10`, cosine annealing applied.

If you set `--epochs ≤ 5` the backbone stays frozen for the entire run (a warning is logged).

## On a SLURM cluster

```bash
sbatch scripts/train_classifier.slurm
```

Edit `#SBATCH --partition` and `pytorch-cuda` version in `environment_gpu.yml` to match your cluster before submitting.
