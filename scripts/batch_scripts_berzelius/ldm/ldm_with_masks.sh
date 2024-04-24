#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:00:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out
# SBATCH --signal=SIGUSR1@90

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

python /proj/aicell/users/x_emmku/stable-diffusion/main.py -t -b /proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_test_masks.yaml -l /proj/aicell/data/stable-diffusion/mca --gpus=0, --seed=123 --scale_lr=False