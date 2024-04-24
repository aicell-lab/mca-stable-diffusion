#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 00:10:00
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2


python /proj/aicell/users/x_emmku/stable-diffusion/main.py -t -b /proj/aicell/users/x_emmku/stable-diffusion/configs/autoencoder/autoencoder_vq_f4_mca_data.yaml -l /proj/aicell/data/stable-diffusion/mca --gpus=1