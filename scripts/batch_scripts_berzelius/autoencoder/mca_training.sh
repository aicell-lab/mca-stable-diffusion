#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
# SBATCH -t 2-10:00:00
#SBATCH -t 00:25:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out
# SBATCH --signal=SIGUSR1@90


module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2


python /proj/aicell/users/x_emmku/stable-diffusion/main.py -t -b /proj/aicell/users/x_emmku/stable-diffusion/mca/models/autoencoder_vq_f4_mca.yaml -l /proj/aicell/data/stable-diffusion/mca --gpus=0,1,2,3