#!/bin/bash
#SBATCH --job-name=hbd_classifier
#SBATCH --output=/home/rg5022/Humming_Bird_Detection/logs/train_classifier_%j.log
#SBATCH --error=/home/rg5022/Humming_Bird_Detection/logs/train_classifier_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_628_general
#SBATCH --job-name=HBD

export WANDB_API_KEY=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

# ── environment ──────────────────────────────────────────────────────────────
module purge

singularity exec --nv \ 
    --overlay /scratch/rg5022/images/rva.ext3:ro \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    bash -c \
"
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate hbd

# ── project root ──────────────────────────────────────────────────────────────
cd /scratch/rg5022/Humming_Bird_Detection

# ── run ───────────────────────────────────────────────────────────────────────
python scripts/train_classifier.py \
    --train-dir  data/interim/train_data/train \
    --val-dir    data/interim/train_data/val \
    --epochs     50 \
    --batch-size 128 \
    --lr         1.2e-3 \
    --weight-decay 1e-4 \
    --early-stopping-patience 5
