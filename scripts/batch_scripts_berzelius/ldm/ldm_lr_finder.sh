#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-02:30:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out
# SBATCH --signal=SIGUSR1@90

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

config="/proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_ldm_version_2.yaml"
logdir="/proj/aicell/data/stable-diffusion/mca"


python /proj/aicell/users/x_emmku/stable-diffusion/main.py -b $config -l $logdir  --gpus=0, --seed=123 --scale_lr=False --no-test=True --use_lr_finder=True --train=False