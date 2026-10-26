#!/bin/bash
#SBATCH --job-name=hbd_seed_sweep
#SBATCH --output=/scratch/rg5022/Humming_Bird_Detection/logs/train_classifier_%A_%a.log
#SBATCH --error=/scratch/rg5022/Humming_Bird_Detection/logs/train_classifier_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_628_general
#SBATCH --array=0-9

# Submit with:
#   export WANDB_API_KEY=$(cat ~/.wandb_api_key) && sbatch scripts/train_classifier_seed_sweep.slurm
# Never commit secrets to the repo.

export WANDB_API_KEY=$(cat ~/.wandb_api_key)

# Map array index to seed
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ── environment ──────────────────────────────────────────────────────────────
module purge

# Ensure log directory exists; SLURM cannot create missing parent directories.
mkdir -p /scratch/rg5022/Humming_Bird_Detection/logs

singularity exec --nv \
    --overlay /scratch/rg5022/images/rva.ext3:ro \
    --env SEED=$SEED \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    bash -c "
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate hbd

cd /scratch/rg5022/Humming_Bird_Detection

python scripts/train_classifier.py \
    --train-dir  data/interim/train_data/train \
    --val-dir    data/interim/train_data/val \
    --epochs     50 \
    --batch-size 128 \
    --lr         1.2e-3 \
    --weight-decay 1e-4 \
    --early-stopping-patience 5 \
    --seed       $SEED
"