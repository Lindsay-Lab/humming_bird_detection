#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_628_general
#SBATCH --job-name=preproc_yolo
#SBATCH -o /scratch/xs2682/hb-yolo/humming_bird_detection/logs/yolo/preprocess_%j.out
#SBATCH -e /scratch/xs2682/hb-yolo/humming_bird_detection/logs/yolo/preprocess_%j.err

cd /scratch/xs2682/hb-yolo/humming_bird_detection

singularity exec --nv --overlay /scratch/xs2682/hb-yolo/overlay-50G-10M.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate hb-yolo; python -m humming_bird_detection.preprocessing.preprocessor --workers 8 --batch-size 16"
