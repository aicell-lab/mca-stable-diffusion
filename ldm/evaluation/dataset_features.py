"""We manage all the idiosyncracies of the original dataset's structure here"""

import itertools
import os
import numpy as np
np.bool = bool
from collections import namedtuple
import mahotas as mt
import pandas as pd
import hashlib
from PIL import Image
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import hpacellseg.cellsegmentator as cellsegmentor
from hpacellseg.utils import label_cell, label_nuclei
from collections.abc import Iterable


"""
`Feature` is basically a class.

name of the feature: index you want in the pandas dataframe
get_feature: function that takes a sample from a dataset and returns the feature
countable: boolean that tells us if it makes sense to count the feature
multiple: boolean that tells us if the feature can simultaneously have multiple values and should be explanded to a one-hot encoding
single_cell: basically should you run get_feature over the cell masks from the segmented image
"""
Feature = namedtuple("feature", ["name", "get_feature", "countable", "multiple", "single_cell"])

def expand_feature(feature, value):
    return f"{feature.name}: \"{value}\""

def get_intensity_mean(sample):
    return np.mean(sample['image'])

def get_intensity_var(sample):
    return np.var(sample['image'])

def normalized_to_uint8(image):
    """Normalize an image to [0, 255]"""
    image -= np.min(image)
    image /= np.max(image)
    image *= 255
    return image.astype(np.uint8)

# def normalized_to_0_1(image):
#     """Normalize an image to [0, 1]"""
#     image -= np.min(image)
#     image /= np.max(image)
#     return image

def get_image_haralick(sample):
    # angular second moment, contrast, correlation, sum of squares: variance, 
    # inverse difference moment, sum average, sum variance, sum entropy, entropy, 
    # difference variance, difference entropy, information measures of correlation 1, 
    # information measures of correlation 2
    image = normalized_to_uint8(sample['image'])
    textures = mt.features.haralick(image) # 13 arrays since 13 3D directions (not 14?)
    ht_mean = textures.mean(axis=0)
    return ht_mean

# Slows down pandas too much
def get_image(sample):
    return sample['image']

def get_4_channel_image(sample):
    poi = sample['image'][:, :, 1].reshape(256, 256, 1)
    return np.concatenate((poi, sample['ref-image']), axis=2)

def get_ref_channel_image(sample):
    return sample['ref-image']

get_image = get_4_channel_image

def get_image_hash(sample):
    image_hash = hashlib.sha1(get_image(sample).tobytes()).hexdigest()
    return image_hash

def get_protein(sample):
    return sample['condition_caption'].split('/')[0]

def get_cell_line(sample):
    return sample['condition_caption'].split('/')[1]

def get_location(sample):
    return sample['location_caption'].split(',')

def get_segmented_cells(samples):
    # torch.nn.Module.dump_patches = True
    segmentor = cellsegmentor.CellSegmentator(
        nuclei_model="/data/xikunz/stable-diffusion/nuclei_model.pth",
        cell_model="/data/xikunz/stable-diffusion/cell_model.pth",
        # recommended 0.25; our samples are already downsized by 0.125
        scale_factor=1,
        # NOTE: setting padding=True seems to solve most issues that have been encountered
        #       during our single cell Kaggle challenge.
        device="cpu",
        padding=True,
        multi_channel_model=True
    )

    # To fix a pytorch 1.11.0 compatibility issue using the solution from https://github.com/ultralytics/yolov5/issues/6948#issuecomment-1299936326
    for m in itertools.chain(segmentor.nuclei_model.modules(), segmentor.cell_model.modules()):
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    bulk_imgs = [normalized_to_uint8(get_ref_channel_image(sample)) for sample in samples]
    # dapi_images = [i[:, :, 2] for i in bulk_imgs]
    nuc_segmentations = segmentor.pred_nuclei(bulk_imgs)
    # nuc_segmentations = segmentor.pred_nuclei(dapi_images)
    cell_segmentations = segmentor.pred_cells(bulk_imgs, precombined=True)
    masks = [label_cell(nuc, cell) for nuc, cell in zip(nuc_segmentations, cell_segmentations)]
    # nuc_masks = [label_nuclei(nuc) for nuc in nuc_segmentations]

    nuc_masks, cell_masks = [], []
    for i in range(len(masks)):
        nuc_masks.append([])
        cell_masks.append([])
        for k in range(1, 1 + masks[i][0].max()):
            nuc_masks[i].append((masks[i][0] == k).reshape(256, 256, 1) * bulk_imgs[i])
            cell_masks[i].append((masks[i][1] == k).reshape(256, 256, 1) * bulk_imgs[i])
            nuc_masks[i][-1] = torch.from_numpy(nuc_masks[i][-1])
            cell_masks[i][-1] = torch.from_numpy(cell_masks[i][-1])

    return cell_masks, nuc_masks


def get_percent_dark(sample):
    img = get_image(sample)
    return np.sum(img[:, :, 1] < 0.1) / (img.shape[0] * img.shape[1])


def get_nuc_cyto(sample):
    ref_img = sample["ref-image"]
    assert ref_img.min() == -1 and ref_img.max() == 1
    cell_masks, nuclei_masks = get_segmented_cells([sample])
    print(f"{len(cell_masks[0])} cells and {len(nuclei_masks[0])} nuclei segmented for image {sample['hpa_index']}!")
    cell_masks = np.stack(cell_masks[0])[:, :, :, 0]
    nuclei_masks = np.stack(nuclei_masks[0])[:, :, :, 0]
    cyto_masks = cell_masks - nuclei_masks
    expression_dist = np.array(list(zip(np.sum(nuclei_masks, axis=(1, 2)), np.sum(cyto_masks, axis=(1, 2)))))
    expression_dist = expression_dist / np.sum(cell_masks, axis=(1, 2)).reshape(-1, 1)
    if np.isnan(expression_dist.flatten()).any():
        expression_dist[np.isnan(expression_dist)] = 0
    return expression_dist


# These are all objects of class `Feature`.
image = Feature("Image", get_image, False, False, False)
img_hash = Feature("Image Hash", get_image_hash, False, False, False)
int_mean = Feature("Intensity Mean", get_intensity_mean, False, False, False)
int_var = Feature("Intensity Var", get_intensity_var, False, False, False)
img_haralick = Feature("Haralick Features", get_image_haralick, False, False, False)
protein = Feature("Protein", get_protein, True, False, False) # arguable, but too many proteins to one-hot encode
cell_line = Feature("Cell Line", get_cell_line, True, False, False)
location = Feature("Location", get_location, True, True, False)
nuc_cyto = Feature("Nuclei & Cytoplasm Expression", get_nuc_cyto, False, False, True)


# TODO: untested as far as missing/features
# Tested on missing images from the cache
# TODO: untested on subset of features from cache entry
# Tries to find sample in cache
# If it's not there it computes them all and then adds an entry
# If it's missing features for that image, it computes them and adds them to the entry
# Always returns a complete list of features in the order of the features list
# TODO: needs a command to refresh cache, for example if feature code was wrong
# Just maintains a picked pandas dataframe so loading speed/size might become a bottle neck
# If it really gets too big, we would need to use a database and expose it as a service maybe
def get_features(samples, features, cache, logdir, dataset_name="", recompute=[]): #, debugging=[image_sampler, segmentation_sampler]):
    if not cache:
        feature_values = []
        for sample in samples:
            sample_feature_values = dict()
            ref_img = sample["ref-image"]
            assert ref_img.min() == -1 and ref_img.max() == 1
            for f in features:
                sample_feature_values[f.name] = f.get_feature(sample)
            feature_values.append(sample_feature_values)
        return feature_values
        # return [{f.name: f.get_feature(sample) for f in features} for sample in samples]

    if logdir is None:
        raise ValueError("logdir must be specified if cache is True")

    if img_hash not in features:
        raise ValueError("Image hash must be in features to lookup cache entries.")

    log_home = "/".join(logdir.split('/')[:-1])
    cache_path = os.path.join(log_home, "feature_cache.pkl")

    df = pd.DataFrame(columns=[f.name for f in features])
    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)

    sample_features = []
    changed = False
    for sample in samples:
        feature_values, sample_changed, df = retrieve_from_cache(sample, features, df, recompute)
        sample_features.append(feature_values)
        changed |= sample_changed

    # images = [torch.from_numpy(get_image(sample)) for sample in samples[:debug_count]]
    images = [torch.from_numpy(get_image(sample)) for sample in samples]
    plot_image_sample(images, dataset_name, logdir, force_rgb=True)

    # cell_masks, nuc_masks = get_segmented_cells(samples[:debug_count])
    cell_masks, nuc_masks = get_segmented_cells(samples)
    plot_segmented_cells(cell_masks, nuc_masks, images, dataset_name, logdir)

    if changed:
        with open(cache_path, "wb") as f:
            df.to_pickle(f)

    return sample_features


# TODO: can store digest of the feature code in the cache to detect changes and automatically recompute
# then we can just get rid of this recompute parameter
def retrieve_from_cache(sample, features, df, recompute=[]):
    feature_values = {}
    key = img_hash.get_feature(sample)
    changed = False
    if key not in df[img_hash.name].values:
        feature_values = {f.name: f.get_feature(sample) for f in features}
        df = df.append(feature_values, ignore_index=True)
    else:
        for f in features:
            if f.name not in df.columns or f in recompute:
                feature_values[f.name] = f.get_feature(sample)
                changed = True
            else:
                val = df.loc[df[img_hash.name] == key, f.name].values[0]
                use_any = type(val) in [np.ndarray, list]
                if (use_any and not pd.isna(val).any()) or (not use_any and not pd.isna(val)):
                    feature_values[f.name] = val
                else:
                    feature_values[f.name] = f.get_feature(sample)
                    changed = True

        # the below will account for new columns too
        if changed == True:
            df.drop(np.argmax([np.array(df[img_hash.name] == key)]), inplace=True)
            df = df.append(feature_values, ignore_index=True)

    return feature_values, changed, df


def cmyk_to_rgb(img, chw=True):
    if not chw:
        return NotImplementedError("Only implemented for chw")
    rgb_img = torch.zeros((3, img.shape[1], img.shape[2]))
    # first change the image to the desired RGB colors
    # then change the encoding
    # key should become blue
    # cyan should become green
    # yellow should stay yellow
    # magenta should become red
    translated = torch.zeros_like(img)
    translated[0] = (img[3] + img[0]) / 2
    translated[1] = (img[3] + img[1]) / 2
    translated[2] = (img[0] + img[1] + img[2]) / 3
    translated[3] = -1 * torch.sum(img, dim=0) / 4
    # convert translated from cmyk to rgb
    rgb_img[0] = translated[1] + translated[2]
    rgb_img[1] = translated[0] + translated[2]
    rgb_img[2] = translated[0] + translated[1]
    rgb_img /= 2
    return rgb_img


# Primarily for debugging things like segmentation, etc.
# Will be called from feature methods that work on these images
def plot_image_sample(images, dataset_name, logdir, force_rgb=False):
    # images start: 256 X 256 X 3: h,w,c
    num_channels = images[0].shape[2]
    img_path = os.path.join(logdir, f'{dataset_name}_sample.tiff')
    grid_images = []
    for image in images:
        img = image.permute(2, 0, 1)
        for i in range(num_channels):
            next_channel = torch.zeros_like(img) - 1
            next_channel[i] = img[i]
            if force_rgb:
                next_channel = cmyk_to_rgb(next_channel)
            grid_images.append(next_channel)
        if force_rgb:
            img = cmyk_to_rgb(img)
        grid_images.append(img)
    grid = make_grid(grid_images, nrow=num_channels + 1, normalize=True)
    grid = grid.permute(1, 2, 0)  # c,h,w -> h,w,c
    grid = grid.numpy()
    grid = normalized_to_uint8(grid)
    Image.fromarray(grid, "RGB" if num_channels == 3 or force_rgb else "CMYK").save(img_path)


def plot_segmented_cells(cell_masks, nuc_masks, images, dataset_name, logdir):
    # images start: 256 X 256 X 3: h,w,c
    max_cells = max([len(cells) for cells in cell_masks])
    img_path = os.path.join(logdir, f'{dataset_name}_segmented.tiff')
    grid_images = []
    assert len(images) == len(cell_masks)
    assert len(nuc_masks) == len(cell_masks)
    images = [255 * (image[:, :, 1:] + 1) / 2 for image in images]
    blank = torch.zeros_like(images[0]) - 1
    for i in range(len(images)):
        grid_images.append(images[i].permute(2, 0, 1))
        for j in range(max_cells):
            next_img = cell_masks[i][j] if j < len(cell_masks[i]) else blank
            next_img = next_img.permute(2, 0, 1)
            grid_images.append(next_img)
        grid_images.append(blank.permute(2, 0, 1))
        for j in range(max_cells):
            next_img = nuc_masks[i][j] if j < len(nuc_masks[i]) else blank
            next_img = next_img.permute(2, 0, 1)
            grid_images.append(next_img)
    grid = make_grid(grid_images, nrow=1 + max_cells, normalize=True)
    grid = grid.permute(1, 2, 0)  # c,h,w -> h,w,c
    grid = grid.numpy()
    grid = normalized_to_uint8(grid)
    Image.fromarray(grid, "RGB").save(img_path)


def get_feature_values(feature, dataset, filter):
    feature_values = dataset[feature.name][filter].values.tolist()
    if feature.multiple:
        raise NotImplementedError("Multiple features 'heterogenous element lengths' not yet implemented")
    elif not isinstance(feature_values[0], Iterable):
        return np.array(feature_values)
    elif feature.single_cell: # inner dim is not relevant
        return np.concatenate(feature_values)
    else:
        return np.stack(feature_values)
    # except ValueError:
    #     V = np.stack([
    #         dataset[feature.name][filter].values.tolist()
    #     ])
