#!/usr/bin/env python3
"""Generate a training report markdown page from a run's saved wandb data.

Usage (from project root)::

    python docs/scripts/generate_training_report.py hummingbird_classifier_20260430_193523

Reads ``models/<stem>_metadata.json``, ``<stem>_wandb_model_metrics.csv``,
``<stem>_wandb_system_metrics.csv``, and ``<stem>_wandb_output.log``.
Produces figures in ``docs/assets/images/training_reports/`` and a
markdown report at ``docs/training-reports/<stem>.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_FIGS = PROJECT_ROOT / "docs" / "assets" / "images" / "training_reports"
OUTPUT_MD = PROJECT_ROOT / "docs" / "training-reports"

DARK_GREEN = "#1b5e20"
GREEN = "#2e7d32"
LIGHT_GREEN = "#c8e6c9"
BG = "#f8faf8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a training report from saved wandb data."
    )
    p.add_argument(
        "run_stem",
        help="Model file stem (e.g. hummingbird_classifier_20260430_193523)",
    )
    p.add_argument(
        "--models-dir",
        default=MODELS_DIR,
        type=Path,
        help="Directory containing model output files",
    )
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────


def load_run_data(run_stem: str, models_dir: Path) -> dict[str, Any]:
    """Load all run data files and return them in a dict.

    Returns keys: metadata (dict), model_metrics (DataFrame),
    system_metrics (DataFrame), output_log (str).
    """
    base = models_dir / run_stem

    with open(base.with_name(f"{run_stem}_metadata.json")) as f:
        metadata = json.load(f)

    model_metrics = pd.read_csv(
        base.with_name(f"{run_stem}_wandb_model_metrics.csv")
    )

    system_metrics = pd.read_csv(
        base.with_name(f"{run_stem}_wandb_system_metrics.csv")
    )

    log_path = base.with_name(f"{run_stem}_wandb_output.log")
    output_log = log_path.read_text() if log_path.exists() else ""

    return {
        "metadata": metadata,
        "model_metrics": model_metrics,
        "system_metrics": system_metrics,
        "output_log": output_log,
    }


# ── plotting functions ────────────────────────────────────────────────────


def _set_style(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=10, fontweight="bold", color=DARK_GREEN, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#666666", labelsize=8)


def _save_fig(fig: plt.Figure, name: str) -> str:
    rel_path = f"../assets/images/training_reports/{name}"
    abs_path = OUTPUT_FIGS / name
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(abs_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rel_path


def plot_loss_accuracy(model_metrics: pd.DataFrame) -> str:
    """Dual-panel: train/val loss (left) and train/val accuracy (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")

    epochs = model_metrics["_step"].astype(int)
    best_epoch_mask = model_metrics["val/is_best"].astype(bool)

    # Loss
    ax1.plot(epochs, model_metrics["train/loss"], color=GREEN, lw=1.5, label="Train")
    ax1.plot(epochs, model_metrics["val/loss"], color="#e53935", lw=1.5, label="Val")
    best = model_metrics[best_epoch_mask]
    ax1.axvline(
        best["_step"].iloc[0], color="#888888", ls="--", lw=1,
        label=f"Best (epoch {int(best['_step'].iloc[0])})",
    )
    ax1.set_xlabel("Epoch", fontsize=8, color="#666666")
    ax1.set_ylabel("Loss", fontsize=8, color="#666666")
    ax1.legend(fontsize=7, frameon=False)
    _set_style(ax1, "Loss")

    # Accuracy
    ax2.plot(epochs, model_metrics["train/accuracy"], color=GREEN, lw=1.5, label="Train")
    ax2.plot(epochs, model_metrics["val/accuracy"], color="#e53935", lw=1.5, label="Val")
    ax2.axvline(best["_step"].iloc[0], color="#888888", ls="--", lw=1)
    ax2.set_xlabel("Epoch", fontsize=8, color="#666666")
    ax2.set_ylabel("Accuracy", fontsize=8, color="#666666")
    ax2.legend(fontsize=7, frameon=False)
    _set_style(ax2, "Accuracy")

    fig.tight_layout()
    return _save_fig(fig, "loss_accuracy.png")


def plot_f1_precision_recall(model_metrics: pd.DataFrame) -> str:
    """Validation precision, recall, and F1 score over epochs."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("white")

    epochs = model_metrics["_step"].astype(int)

    ax.plot(epochs, model_metrics["val/precision"], color="#1565c0", lw=1.5, label="Precision")
    ax.plot(epochs, model_metrics["val/recall"], color="#e65100", lw=1.5, label="Recall")
    ax.plot(epochs, model_metrics["val/f1"], color=DARK_GREEN, lw=2.0, label="F1")

    best = model_metrics[model_metrics["val/is_best"].astype(bool)]
    ax.axvline(
        best["_step"].iloc[0], color="#888888", ls="--", lw=1,
        label=f"Best (epoch {int(best['_step'].iloc[0])})",
    )

    ax.set_xlabel("Epoch", fontsize=8, color="#666666")
    ax.set_ylabel("Score", fontsize=8, color="#666666")
    ax.set_ylim(0.6, 1.01)
    ax.legend(fontsize=7, frameon=False)
    _set_style(ax, "Validation: Precision / Recall / F1")

    fig.tight_layout()
    return _save_fig(fig, "precision_recall_f1.png")


def plot_learning_rate(model_metrics: pd.DataFrame) -> str:
    """Learning rate schedule over epochs."""
    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_facecolor("white")

    epochs = model_metrics["_step"].astype(int)
    ax.plot(epochs, model_metrics["train/lr"], color=GREEN, lw=1.5, marker="o", ms=3)
    ax.set_xlabel("Epoch", fontsize=8, color="#666666")
    ax.set_ylabel("Learning rate", fontsize=8, color="#666666")
    _set_style(ax, "Learning rate schedule")

    fig.tight_layout()
    return _save_fig(fig, "learning_rate.png")


def plot_system_gpu(system_metrics: pd.DataFrame) -> str:
    """GPU utilization, memory, and temperature over time."""
    df = system_metrics.copy()
    df["min_rt"] = df["_runtime"] - df["_runtime"].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.3)

    # GPU utilization %
    ax = axes[0]
    ax.fill_between(df["min_rt"], df["system.gpu.0.gpu"], alpha=0.3, color=GREEN)
    ax.plot(df["min_rt"], df["system.gpu.0.gpu"], color=GREEN, lw=1.0)
    ax.set_ylabel("GPU util %", fontsize=8, color="#666666")
    ax.set_ylim(0, 100)
    _set_style(ax, "GPU utilization")

    # GPU memory %
    ax = axes[1]
    ax.fill_between(df["min_rt"], df["system.gpu.0.memory"], alpha=0.3, color="#1565c0")
    ax.plot(df["min_rt"], df["system.gpu.0.memory"], color="#1565c0", lw=1.0)
    ax.set_ylabel("GPU mem %", fontsize=8, color="#666666")
    ax.set_ylim(0, 100)
    _set_style(ax, "GPU memory")

    # GPU temperature
    ax = axes[2]
    ax.fill_between(df["min_rt"], df["system.gpu.0.temp"], alpha=0.3, color="#e65100")
    ax.plot(df["min_rt"], df["system.gpu.0.temp"], color="#e65100", lw=1.0)
    ax.set_xlabel("Runtime (s)", fontsize=8, color="#666666")
    ax.set_ylabel("GPU temp °C", fontsize=8, color="#666666")
    _set_style(ax, "GPU temperature")

    fig.tight_layout()
    return _save_fig(fig, "system_gpu.png")


def plot_system_cpu_ram(system_metrics: pd.DataFrame) -> str:
    """CPU utilization, process memory RSS, and disk usage over time."""
    df = system_metrics.copy()
    df["min_rt"] = df["_runtime"] - df["_runtime"].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.3)

    # CPU %
    ax = axes[0]
    ax.fill_between(df["min_rt"], df["system.cpu"], alpha=0.3, color=GREEN)
    ax.plot(df["min_rt"], df["system.cpu"], color=GREEN, lw=1.0)
    ax.set_ylabel("CPU %", fontsize=8, color="#666666")
    _set_style(ax, "System CPU utilization")

    # Process memory RSS
    ax = axes[1]
    rss_gb = df["system.proc.memory.rssMB"] / 1024
    ax.fill_between(df["min_rt"], rss_gb, alpha=0.3, color="#7b1fa2")
    ax.plot(df["min_rt"], rss_gb, color="#7b1fa2", lw=1.0)
    ax.set_ylabel("RSS (GB)", fontsize=8, color="#666666")
    _set_style(ax, "Process memory (RSS)")

    # Disk usage %
    ax = axes[2]
    disk_col = None
    for c in df.columns:
        if "disk" in c and "usagePercent" in c:
            disk_col = c
            break
    if disk_col:
        ax.fill_between(df["min_rt"], df[disk_col], alpha=0.3, color="#00838f")
        ax.plot(df["min_rt"], df[disk_col], color="#00838f", lw=1.0)
    ax.set_xlabel("Runtime (s)", fontsize=8, color="#666666")
    ax.set_ylabel("Disk %", fontsize=8, color="#666666")
    _set_style(ax, "Disk usage")

    fig.tight_layout()
    return _save_fig(fig, "system_cpu_ram.png")


# ── markdown builders ─────────────────────────────────────────────────────


def _bytes_to_gb(b: int | str) -> str:
    b = int(b)
    return f"{b / 1e9:.1f} GB"


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=None).strftime("%Y-%m-%d %H:%M:%S")


def build_report(
    run_stem: str,
    metadata: dict,
    model_metrics: pd.DataFrame,
    system_metrics: pd.DataFrame,
    output_log: str,
    figure_paths: dict[str, str],
) -> str:
    """Compile the markdown report."""
    hp = metadata["hyperparameters"]
    res = metadata["results"]
    wb = metadata.get("wandb", {})
    wb_meta = wb.get("metadata", {}).get("_metadata", {})
    slurm = wb_meta.get("slurm", {})
    gpu_info = wb_meta.get("gpu_nvidia", [{}])[0] if wb_meta.get("gpu_nvidia") else {}

    best_epoch = int(res["best_epoch"])
    best_row = model_metrics[model_metrics["_step"] == best_epoch]
    best = best_row.iloc[0] if len(best_row) > 0 else model_metrics.iloc[-1]

    runtime_s = system_metrics["_runtime"].iloc[-1] - system_metrics["_runtime"].iloc[0]
    runtime_m, runtime_s = divmod(int(runtime_s), 60)

    # Training log summary — extract per-epoch lines
    epoch_lines = [
        ln for ln in output_log.splitlines()
        if "Epoch" in ln and "train_loss" in ln
    ]

    # Build markdown
    lines: list[str] = []
    lines.append(f"# Training Report — {run_stem}")
    lines.append("")
    lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**W&B Run:** [{wb.get('run_id', '?')}]({wb.get('run_url', '#')})  \n"
        f"**Project:** `{wb.get('project', '?')}`  \n"
        f"**Git commit:** `{metadata.get('git_commit', '?')[:8]}`  \n"
        f"**SLURM Job ID:** {slurm.get('job_id', metadata.get('slurm_job_id', 'N/A'))}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Compute Resources ──
    if wb_meta:
        lines.append("## Compute Resources")
        lines.append("")
        lines.append("| Resource | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Host | `{wb_meta.get('host', '?')}` |")
        lines.append(f"| GPU | {wb_meta.get('gpu', '?')} ×{wb_meta.get('gpu_count', '?')} |")
        if gpu_info:
            lines.append(f"| GPU memory | {_bytes_to_gb(gpu_info.get('memoryTotal', 0))} |")
            lines.append(f"| CUDA cores | {gpu_info.get('cudaCores', '?')} |")
            lines.append(f"| Architecture | {gpu_info.get('architecture', '?')} |")
        lines.append(f"| CPU cores | {wb_meta.get('cpu_count', '?')} |")
        if wb_meta.get("memory", {}).get("total"):
            lines.append(f"| RAM | {_bytes_to_gb(wb_meta['memory']['total'])} |")
        if slurm:
            lines.append(f"| Partition | `{slurm.get('job_partition', '?')}` |")
            lines.append(f"| QoS | `{slurm.get('job_qos', '?')}` |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Hyperparameters ──
    lines.append("## Hyperparameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Epochs (requested) | {hp['epochs']} |")
    lines.append(f"| Epochs (completed) | {len(model_metrics)} |")
    lines.append(f"| Batch size | {hp['batch_size']} |")
    lines.append(f"| Learning rate | {hp['lr']} |")
    lines.append(f"| Weight decay | {hp['weight_decay']} |")
    lines.append(f"| Early stopping patience | {hp['early_stopping_patience']} |")
    lines.append(f"| Seed | {hp['seed']} |")
    lines.append(f"| Train dir | `{metadata.get('data', {}).get('train_dir', '?')}` |")
    lines.append(f"| Val dir | `{metadata.get('data', {}).get('val_dir', '?')}` |")
    lines.append("")

    # ── Best Result ──
    lines.append("## Best Result")
    lines.append("")
    lines.append("| Metric | Value | Epoch |")
    lines.append("|--------|-------|-------|")
    lines.append(f"| Best val loss | {res['best_val_loss']:.4f} | {int(res['best_epoch'])} |")
    lines.append(
        f"| Val accuracy at best | {best['val/accuracy']:.4f} | {int(best['_step'])} |"
    )
    lines.append(
        f"| Val precision at best | {best['val/precision']:.4f} | {int(best['_step'])} |"
    )
    lines.append(
        f"| Val recall at best | {best['val/recall']:.4f} | {int(best['_step'])} |"
    )
    lines.append(
        f"| Val F1 at best | {best['val/f1']:.4f} | {int(best['_step'])} |"
    )
    lines.append(f"| Training duration | {runtime_m}m {runtime_s}s | — |")
    lines.append("")

    # ── Final Epoch Metrics ──
    final = model_metrics.iloc[-1]
    lines.append("## Final Epoch Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Train loss | {final['train/loss']:.4f} |")
    lines.append(f"| Train accuracy | {final['train/accuracy']:.4f} |")
    lines.append(f"| Val loss | {final['val/loss']:.4f} |")
    lines.append(f"| Val accuracy | {final['val/accuracy']:.4f} |")
    lines.append(f"| Val precision | {final['val/precision']:.4f} |")
    lines.append(f"| Val recall | {final['val/recall']:.4f} |")
    lines.append(f"| Val F1 | {final['val/f1']:.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Model Metrics Figures ──
    lines.append("## Model Metrics")
    lines.append("")
    lines.append(f"![Loss and Accuracy]({figure_paths['loss_accuracy']})")
    lines.append("")
    lines.append(f"![Precision, Recall, F1]({figure_paths['f1']})")
    lines.append("")
    lines.append(f"![Learning Rate]({figure_paths['lr']})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── System Metrics Figures ──
    lines.append("## System Metrics")
    lines.append("")
    lines.append(f"![GPU]({figure_paths['gpu']})")
    lines.append("")
    lines.append(f"![CPU, RAM, Disk]({figure_paths['cpu_ram']})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per-Epoch Log ──
    lines.append("## Per-Epoch Training Log")
    lines.append("")
    if epoch_lines:
        lines.append("```text")
        for ln in epoch_lines:
            lines.append(ln.strip())
        lines.append("```")
    else:
        lines.append("_No per-epoch log lines found in output._")
    lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    print(f"Generating training report for: {args.run_stem}")
    print(f"  Data dir : {args.models_dir}")
    print()

    # Check files exist
    base = args.models_dir / args.run_stem
    required = [
        base.with_name(f"{args.run_stem}_metadata.json"),
        base.with_name(f"{args.run_stem}_wandb_model_metrics.csv"),
        base.with_name(f"{args.run_stem}_wandb_system_metrics.csv"),
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Load data
    data = load_run_data(args.run_stem, args.models_dir)
    model_metrics = data["model_metrics"]
    system_metrics = data["system_metrics"]

    sns.set_theme(style="whitegrid", context="notebook")

    # Generate figures
    figure_paths: dict[str, str] = {}
    figure_paths["loss_accuracy"] = plot_loss_accuracy(model_metrics)
    figure_paths["f1"] = plot_f1_precision_recall(model_metrics)
    figure_paths["lr"] = plot_learning_rate(model_metrics)
    figure_paths["gpu"] = plot_system_gpu(system_metrics)
    figure_paths["cpu_ram"] = plot_system_cpu_ram(system_metrics)

    # Build and save report
    report = build_report(
        run_stem=args.run_stem,
        metadata=data["metadata"],
        model_metrics=model_metrics,
        system_metrics=system_metrics,
        output_log=data["output_log"],
        figure_paths=figure_paths,
    )

    report_path = OUTPUT_MD / f"{args.run_stem}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    print(f"Report saved: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Figures saved to: {OUTPUT_FIGS.relative_to(PROJECT_ROOT)}/")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
