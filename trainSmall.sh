#!/bin/bash
#SBATCH --job-name=train_INF_NOISE
#SBATCH --partition=gpu
#SBATCH --ntasks=64
#SBATCH --nodes=2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

conda activate QUYET_NOISE

python ./preprocessing/negative_samples.py