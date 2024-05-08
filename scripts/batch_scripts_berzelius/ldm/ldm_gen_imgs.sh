#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-20:00:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

# args for mca_diffusion_sample parser that won't change that often
ckpt="/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/checkpoints/epoch=000002.ckpt"
config="/proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_gen_imgs_debug.yaml"
gpu=0

#args that will change often
scale=$1
steps=$2
comparison=$3
skip_images=$4

echo "Config used: $config"
echo "Ckpt used: $ckpt"
echo "$skip_imgs skip images"
echo "$scale scale"
echo "$steps steps"

python /proj/aicell/users/x_emmku/stable-diffusion/scripts/img_gen/mca_diffusion_sample.py --checkpoint=$ckpt --config=$config --gpu=$gpu --scale=$scale --steps=$steps -f single -k comparison_image=$comparison skip_images=$skip_images