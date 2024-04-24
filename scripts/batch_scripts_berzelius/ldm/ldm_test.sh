#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 3-0:00:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out
# SBATCH --signal=SIGUSR1@90

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

config=$1
echo "Config file: $config"


python /proj/aicell/users/x_emmku/stable-diffusion/main.py -t -b $config -l /proj/aicell/data/stable-diffusion/mca --gpus=0, --seed=123 --scale_lr=False