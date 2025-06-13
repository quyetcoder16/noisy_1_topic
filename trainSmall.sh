#!/bin/bash
#SBATCH --job-name=train_INF_NOISE
#SBATCH --partition=small
#SBATCH --ntasks=64
#SBATCH --nodes=2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

conda activate QUYET_NOISE

python ./train_glie.py