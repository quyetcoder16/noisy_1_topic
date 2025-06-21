#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --partition=gpu
#SBATCH --ntasks=64
#SBATCH --nodes=2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

conda activate QUYET_NOISE

python ./preprocessing/inf_ss.py