# Training Reports

Automatically generated reports for classifier fine-tuning runs. Each report
compiles metadata, hyperparameters, model metrics, system resource usage, and
the per-epoch training log from data recorded by Weights & Biases.

## Available reports

| Run ID | Date | Best Val Loss | Best Epoch | GPU | Duration |
|--------|------|---------------|------------|-----|----------|
| [hummingbird_classifier_20260430_193523](hummingbird_classifier_20260430_193523.md) | 2026-04-30 | 0.0804 | 7 | NVIDIA L40S | 7m 45s |

## Generate a new report

After a training run completes (and the wandb data has been saved alongside
the model checkpoint), generate a report from the project root:

```bash
python docs/scripts/generate_training_report.py hummingbird_classifier_<run_id>
```

This produces:

- `docs/training-reports/<run_id>.md` — the report page
- `docs/assets/images/training_reports/` — figures (loss, accuracy, F1,
  GPU/CPU metrics)

The report is linked here automatically — add a row to the table above to
include it in the navigation.

## Report contents

Each report includes:

| Section | Content |
|---------|---------|
| **Header** | Run ID, W&B link, git commit, SLURM job ID |
| **Compute Resources** | Host, GPU model/memory/cores, CPU, RAM, partition |
| **Hyperparameters** | Requested vs. completed epochs, batch size, LR, seed |
| **Best Result** | Metrics at the epoch with lowest validation loss |
| **Final Epoch Metrics** | All metrics from the last completed epoch |
| **Model Metrics** | Loss/accuracy curves, precision/recall/F1, learning rate |
| **System Metrics** | GPU utilization/memory/temp, CPU, RAM, disk usage |
| **Per-Epoch Log** | Raw training log lines for each epoch |
