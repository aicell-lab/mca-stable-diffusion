#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:30:00
#SBATCH -C 'thin'
#SBATCH -o /proj/aicell/data/stable-diffusion/mca/logs/slurm-%j.out

module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ldm2

# args for mca_diffusion_sample parser that won't change that often
ckpt="/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/checkpoints/epoch=000002.ckpt"
config="/proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_gen_imgs_debug.yaml"


#args that will change often
scale=1
steps=100

name=$1
num_images=$2
variation=$3


echo "Config used: $config"
echo "Ckpt used: $ckpt"
echo "$scale scale"
echo "$steps steps"

if [[ $variation = "mst" ]]; then
    # generate mst variations, from 1 to 19 by increments of 2
    poi="TPR"
    z=16
    for mst in {1..19..2}
    do
        echo "Generating $num_images images with MST=$mst"
        python /proj/aicell/users/x_emmku/stable-diffusion/scripts/img_gen/mca_diffusion_sample.py --checkpoint=$ckpt --config=$config --scale=$scale --steps=$steps --name=$name -f 'conditions' -k num_images=$num_images protein_str=$poi z=$z mst=$mst
    done
elif [[ $variation = "z" ]]; then
    # generate z variations, from 0 to 30 by increments of 3
    poi="TPR"
    mst=10
    for z in {0..30..3}
    do
        echo "Generating $num_images images with z=$z"
        python /proj/aicell/users/x_emmku/stable-diffusion/scripts/img_gen/mca_diffusion_sample.py --checkpoint=$ckpt --config=$config --scale=$scale --steps=$steps --name=$name -f 'conditions' -k num_images=$num_images protein_str=$poi z=$z mst=$mst
    done
else
    mst=10
    z=16
    pois=("APC2" "CENPA" "CTCF" "KIF4A" "MAD2L1" "NCAPD3" "NUP214" "PLK1" "RACGAP1" "RANBP2" "STAG2" "WAPL")
    for poi in ${pois[@]}
    do
        echo "Generating $num_images images with poi=$poi"
        python /proj/aicell/users/x_emmku/stable-diffusion/scripts/img_gen/mca_diffusion_sample.py --checkpoint=$ckpt --config=$config --scale=$scale --steps=$steps --name=$name -f 'conditions' -k num_images=$num_images protein_str=$poi z=$z mst=$mst
    done
fi    

echo "Done generating images stored in folder $name"