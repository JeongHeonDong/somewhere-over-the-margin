#!/bin/sh

#SBATCH -J CV_Project
#SBATCH -p titanxp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --gres=gpu:2

module purge
module load cuda/10.2 conda
conda activate cv-project

srun python -m src.examples.tripletMarginLossMNIST
