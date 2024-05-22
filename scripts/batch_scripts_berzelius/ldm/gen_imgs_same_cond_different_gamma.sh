#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-02:30:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

# args for mca_diffusion_sample parser that won't change that often
ckpt="/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/checkpoints/epoch=000002.ckpt"
config="/proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_gen_imgs_debug.yaml"


#args that will change often
steps=100

name=$1
num_images=$2
mst=$3
z=$4
poi=$5


echo "Config used: $config"
echo "Ckpt used: $ckpt"
echo "$steps steps"
scales=(6 7 8 9 11)

for scale in ${scales[@]}; do
    echo "Generating $num_images images with mst:$mst, z:$z and poi:$poi with scale=$scale"
    python /proj/aicell/users/x_emmku/stable-diffusion/scripts/img_gen/mca_diffusion_sample.py --checkpoint=$ckpt --config=$config --scale=$scale --steps=$steps --name=$name -f 'conditions' -k num_images=$num_images protein_str=$poi z=$z mst=$mst
done

 

echo "Done generating images stored in folder $name in the ckpt directory"