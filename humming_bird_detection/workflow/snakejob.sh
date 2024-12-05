#!/usr/bin/env bash
#SBATCH -o /home/ rg5022/ humming_bird_detection/slurm/%j.out
#SBATCH -e home/ rg5022/ humming_bird_detection/slurm/%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=60G
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=10             # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --job-name=hbd
#SBATCH --mail-type=END,FAIL
# #SBATCH --gres=cpu

conda activate hbd

snakemake --unlock --cores=1
snakemake --jobname '{jobid}.{rulename}' \
	      --latency-wait 90 \
          --keep-going \
          --nolock \
          --rerun-incomplete \
          --cores=1 \
          --jobs=1 