# Project Name

## Description

This project contains code for evaluation of generated images from the mitotic cell atlas model.

## Code Source

Code is based upon functions from the repository fld from https://github.com/marcojira/fld.git.

## Environment Setup

To set up the environment for this project, follow these steps:

1. Create a conda environment using the provided `environment.yaml` file: `conda env create -f environment.yaml`
2. Activate the conda environment: `conda activate eval`

## Analysis

This project supports the following analysis:
- Calculation of metrics for unconditionally generated images (FID, FLD, % authentic samples, precision, recall) using calculate_metrics.py
- A method to evaluate conditionally generated images with average_embedding.py


### Calculate Metrics

To calculate metrics, run the following command: `python fld/calculate_metrics.py -g /path/to/generated/images -t /path/to/training/images -f Inception --cache-path /path/to/cache_the_features`

### Conditional evaluation using average_embedding.py

The method is based upon feature extraction from the images using a feature extractor (DINOv2, CLIP or Inception). Then the features are dimension reduced to two dimensions to visualise them.

There are two primary analyses to perform. One where one label in the MCA dataset varies and the others are fixed and one where the different guidance levels are explored using this method.

#### Different guidance levels
To explore this, run the following command:`python fld/average_embedding.py different_guidance --guide /path/to/directory/with/differnt/guidance/images --dir_regex regex_to_find_subdirs_in_guide --gt /path/to/dir/of/ground/truth --save_path optional/path/to/cache/features --extractor Inception --labelregex regex_for_labels_from_dirnames_optional`


#### Varying one condition
Run the following command: `python fld/average_embedding.py different_conditions --function umap --different_cond z --image_type gen --extractor Inception --dir /path/to/dir/with/subdirs --regex regex_to_find_subdirs_with_images_in_dir --save_path /path/to/cache/features/optional`