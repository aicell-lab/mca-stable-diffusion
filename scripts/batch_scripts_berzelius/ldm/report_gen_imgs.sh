#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-02:10:00  # generation per image is about 12s calculate t based on num images in total you create + extra time for starting
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

# args for mca_diffusion_sample parser that won't change that often
ckpt="/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/checkpoints/epoch=000002.ckpt"
config="/proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_gen_imgs_debug.yaml"


#args that will change often
scale=10
steps=100

name=$1
num_images=$2
variation=$3


echo "Config used: $config"
echo "Ckpt used: $ckpt"
echo "$scale scale"
echo "$steps steps"



echo "Generating $num_images images per $variation"
python /proj/aicell/users/x_emmku/stable-diffusion/scripts/img_gen/mca_diffusion_sample.py --checkpoint=$ckpt --config=$config --scale=$scale --steps=$steps --name=$name -f 'vary_one_cond' -k num_images=$num_images variation=$variation

echo "Done generating images stored in folder $name"