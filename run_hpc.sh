#!/usr/bin/env bash
#SBATCH --job-name=bio_b_tracking
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=16
#SBATCH --time=480          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=32G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate bio-b-env
python run.py --params=resources/params_tracking_hpc.yml