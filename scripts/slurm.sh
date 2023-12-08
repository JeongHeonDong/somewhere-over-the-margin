#!/bin/sh

#SBATCH -J CV_Project
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -t 1-00:00:00

#### Select GPU
#SBATCH -p titanxp
#SBATCH --gres=gpu:2

#### Select node (change nodelist at your circumstances)
#SBATCH --nodelist=n1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

cd  $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge
module load cuda/10.2 conda

echo "Start"

echo "conda activate cv-project"
conda activate cv-project

SAMPLES_DIR=$HOME/somewhere-over-the-margin/
cd $SAMPLES_DIR

### Modify this line when you want other jobs
python -m src.base --activation gelu --trial base5

date

echo "conda deactivate cv-project"
conda deactivate

squeue --job $SLURM_JOBID

echo "##### END #####"
