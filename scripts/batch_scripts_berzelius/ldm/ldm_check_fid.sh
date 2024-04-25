#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 0-1:00:00
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/fid_calculations/slurm-%j.out

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

# calculate FID score between two distributions of images in the two folders given as arguments
# $1 is the path to the ground truth images, $2 is the path to the generated images
ground_truth_imgs=$1
generated_imgs=$2
current_time=$(date "+%Y-%m-%d-%H:%M:%S")

echo "Current time: $current_time"
echo "Ground truth folder path: $ground_truth_imgs"
echo "Generated folder path: $generated_imgs"

python -m pytorch_fid $ground_truth_imgs $generated_imgs