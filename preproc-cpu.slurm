#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --account=csci_ga_3033-2025fa
#SBATCH --partition=interactive
#SBATCH --job-name=preproc
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=xs2682@nyu.edu
#SBATCH --output=preproc-%j.out
#SBATCH --error=preproc-%j.err

set -e

singularity exec --bind /scratch --nv --overlay /scratch/xs2682/overlay-25GB-500K.ext3:rw /scratch/xs2682/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate hb-yolo; cd /scratch/xs2682/humming_bird_detection; echo 'Starting job'; python -m humming_bird_detection.preprocessor"
