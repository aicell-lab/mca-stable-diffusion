
# Mitotic Cell Atlas Stable Diffusion

**IMPORTANT: checkpoints to models are too large for GitHub. Checkpoints for the autoencoder and the LDM are stored on Berzelius**

## How to run
Quick instructions of how to run. 

### Training autoencoder

Train autoencoder on 2 gpus. Change logdir to where images, ckpts will be stored during training. Change -b to another config to train a new model.


````
conda activate ldm2
python main.py -t -b mca/models/autoencoder_vq_f4_mca.yaml -l /path/to/logdir --gpus=0,1
````

### Training LDM
Train LDM on 1 gpu. Change logdir to where images, ckpts and logfiles will be stored during training. Change -b to another config to train a new model.
````
conda activate ldm2
python main.py -t -b mca/models/mca_config.yaml -l path/to/logdir --gpus=0, --scale_lr=False
````


### Sampling LDM
Generate 1000 images with guidance 1 and ddim steps 100 using the trained ldm. Change checkpoint and config to get new model.
````
conda activate ldm2
python mca_diffusion_sample.py --checkpoint=path/to/mca_ldm.ckpt --config=mca/models/mca_config.yaml --steps=100 --scale=1 -f single_random -k num_images=1000
````


### Evaluation
Evaluation pipeline. More descriptions in evaluation/evaluation.md 

Calculation of metrics:
````
conda activate eval
python calculate_metrics.py -g /path/to/generated/images -t /path/to/training/images -f Inception --cache-path /path/to/cache_the_features
````

Feature extraction and UMAP dimension reduction:
````
conda activate eval
python average_embedding.py different_guidance --guide /path/to/directory/with/different/guidance/images --dir_regex regex_to_find_subdirs_in_guide --gt /path/to/dir/of/ground/truth --save_path optional/path/to/cache/features --extractor Inception --labelregex regex_for_labels_from_dirnames_optional
````
````
conda activate eval
python fld/average_embedding.py different_conditions --function umap --different_cond z --image_type gen --extractor Inception --dir /path/to/dir/with/subdirs --regex regex_to_find_subdirs_with_images_in_dir --save_path /path/to/cache/features/optional
````


## Folder structure

#### Configs
General configs utilized sometime during the work. Look here for inspiration about how to include noise in images or use masks.

#### Evaluation
Contains code for the evaluation pipeline. Code is documented and described further how to use in evaluation/evaluation.md.

#### Models
Configs for the autoencoder and ldm to use for training or sampling.  The `ckpt_path` needs to be changed for training. Note that the checkpoints are too large to store on GitHub. 

## Important files 

#### cell_features_necessary_columns.txt
Features that Cai et al have calculated using their mitosis model. Necessary to extract labels to the `MCACombineDataset`.

File is found in mca-stable-diffusion/mca/labels/cell_features_necessary_columns.txt

#### environment2.yaml
Use this file to create a conda/mamba environment to use when training and sampling from the ldm. 

File is found in mca-stable-diffusion/environment2.yaml

#### mca.py
Dataset file to read and process image and label data associated with the Mitotic Cell Atlas. See documentation in file for more implementation details. 

- `MCACombineDataset` dataset class for loading and pre-processing images. Also handles reading of labels. 
- `MCAConditionEmbedder` class for the conditional embedder in the LDM
- `create_cached_paths` function for caching the paths to all the z-stack images. Speeds up the dataloading
- `find_images_with_same_condition` function to get all real images with a specific condition

This file is found under mca-stable-diffusion/ldm/data/mca.py

#### mca_sample_diffusion.py
Script for sampling from the trained latent diffusion model. Can perform many different types of sampling. 

`--checkpoint` model checkpoint to use for generation

`--config` model config to use for generation

`--scale` is the guidance scale and decides how much to consider the conditions when generating. For the current mca model it does not work well. scale=0 corresponds to unconditional generation and anything higher gives conditional generation.

`--steps` are the number of ddim steps to use during sampling. steps=100 have been tested to work well. 

`--name` optional name of directory in the checkpoint directory where images will be saved

`--gpu` optional if you need to use a specific gpu

`-f` which sampling function to use
    - `single` generates `num_images` from the dataset by drawing random conditions and sampling from them. 
    - `conditions` generates `num_images` with the condition `mst`, `z` and `protein_str`.

`-k` sampling function specific conditions. Input on the form: `-k key1=val1 key2=val2 ... keyn=valn`

Generate 1000 images with guidance scale 1 and 100 DDIM steps. 
````
python scripts/img_gen/mca_diffusion_sample.py --checkpoint=path/to/mca_ldm.ckpt --config=mca/models/mca_config.yaml --scale=1 --steps=100 -f single -k num_images=1000
````

Generate 100 images with the condition mst=5, z=14 and protein=CENPA with guidance scale 1 and 100 DDIM steps. All available proteins exist in protein_dict in mca.py. The ranges for mst are integers in [1, 20] and for z it is integers in [0, 30]. Example:
````
python scripts/img_gen/mca_diffusion_sample.py --checkpoint=path/to/mca_ldm.ckpt --config=mca/models/mca_config.yaml --scale=1 --steps=100 -f conditions -k num_images=100 mst=5 z=14 protein_str=CENPA
````

There are many other functions to choose from in mca_diffusion_sample.py, but these are the main ones to generate images and the other functions are written for a specific purpose. File is found in mca-stable-diffusion/scripts/img_gen/mca_sample_diffusion.py


### Berzelius specific info
There are a bunch of batch scripts written for Berzelius. Useful in terms of slurm commands and also example usage. In scripts/batch_scripts_berzelius.

The data on Berzelius is stored in the `mca` folder. There is a README.md there to explain the structure of `mca`. 

