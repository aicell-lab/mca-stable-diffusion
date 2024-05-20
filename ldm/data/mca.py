from aicsimageio.aics_image import AICSImage
from aicsimageio.readers.tiff_reader import TiffReader
from torch.utils.data import Dataset, DataLoader
import glob
import re
import os
import pandas as pd
import sys
import numpy as np
import random
import cv2
from skimage.transform import rotate
import json
import time
import torch.nn as nn
import torch
from ldm.util import instantiate_from_config

# all available proteins from v1.0.1 and v1.1
protein_dict = {'APC2': 0, 'AURKB': 1, 'BUB1': 2, 'BUBR1': 3, 
                'CDCA8': 4, 'CENPA': 5, 'CEP192': 6, 'CEP250': 7, 
                'CTCF': 8, 'KIF11': 9, 'KIF4A': 10, 'MAD2L1': 11, 
                'MIS12': 12, 'NCAPH2': 13, 'NEDD1': 14, 'NES': 15, 
                'NUP107': 16, 'NUP214': 17, 'PLK1': 18, 'RACGAP1': 19,
                'RANBP2': 20, 'SCC1': 21, 'STAG1': 22, 'STAG2': 23, 
                'TOP2A': 24, 'TPR': 25, 'TUBB2C': 26, 'WAPL': 27,
                'NCAPH': 28, 'NCAPD2': 29, 'NCAPD3': 30, 'SMC4': 31 } 

# number of stages, stack indices and proteins in the dataset
size_dict = {0: 20, 1: 31, 2: len(protein_dict)} 

        
class MCACombineDataset(Dataset):
    """Dataset class for the mitotic cell atlas raw images"""
    def __init__(self, directories, cell_features, use_cached_paths=True, cached_paths='', group='train', z_stacks=31, seed=123, train_val_split=0.95, image_path='*/*/rawtif/*.tif',
                 random_angle=False, rotation_mode='reflect', normalization=False, percentiles=(1, 99), add_noise=False,
                 noise_type = 'gaussian', cell_regex = None, return_masks=False, skip_top_bottom=0, *args, **kwargs):
        """Input:
        directories: paths to all relevant directories with the experiment folders. Multiple paths should be separated by any whitespace character
        cell_features: path to cell_features_necessary_columns.txt
        seed and train_val_split to split the dataset into train and val
        image_paths: relative glob style path from directories path to the .tif files
        use_cached_paths: if True use the cached paths to save some time when loading the dataset
        cached_paths: path to the cached paths
        group: 'train' or 'validation' to split the dataset
        z_stacks: number of z-stacks in the images for the channels
        random_angle: if True apply random angle transformation to images
        rotation_mode: mode for scimage.transform.rotate if random_angle
        normalization: if True apply percentile normalization to images for each channel separately
        percentiles: tuple with percentiles to use for normalization if normalization
        add_noise: if True add noise to the images
        noise_type: str with the noise type. Currently Gaussian noise is the only implemented type which adds gaussian with zero mean and std(channel) separately to the images
        cell_regex: regex to match the path column in cell_features.txt
        return_masks: if True return the masks as well. Mask is currently only used with the cell volume mask
        skip_top_bottom: number of top and bottom slices to skip in the z-stack
        """

        super().__init__()
        
        if use_cached_paths:
            assert cached_paths != '', 'You forgot to add the cached paths'
            with open(cached_paths, 'r') as file:
                self.img_paths= json.load(file)
        else:
            directories = directories.split()
            self.img_paths = []
            # get paths to all images
            for d in directories:  
                    self.img_paths.extend(sorted(glob.glob(os.path.join(d, image_path)))) # sort to assure reproducibility no matter the os as glob returns paths in arbitrary order
        
        random.Random(seed).shuffle(self.img_paths)

        # split into train and val based on train_val_split
        assert train_val_split > 0 and train_val_split <= 1
        assert group in ['train', 'validation'] 
        if train_val_split < 1:
            split = int(len(self.img_paths) * train_val_split)
            if group == 'train':
                self.img_paths = self.img_paths[:split]
            else: 
                self.img_paths = self.img_paths[split:]
    
        # get all cell features 
        assert type(skip_top_bottom) == int and type(z_stacks) == int
        self.cell_features = pd.read_csv(cell_features, sep='\s+')
        self.cell_regex = "(\d{6}_[A-Za-z0-9-]+_[A-Za-z12]+\/cell\d+_R\d+)" if cell_regex is None else cell_regex
        self.z_stacks = z_stacks
        self.random_angle = random_angle
        self.rotation_mode = rotation_mode if self.random_angle else None
        self.normalization = normalization
        self.percentiles = percentiles if self.normalization else None
        self.add_noise = add_noise
        self.noise_type = noise_type if self.add_noise else None
        self.return_masks = return_masks
        assert skip_top_bottom >= 0 and skip_top_bottom < self.z_stacks/2, 'skip_top_bottom must be between 0 and z_stacks/2'
        self.skip_top_bottom = skip_top_bottom

    
    def _get_cell_stage_from_path(self, img_path, regex="(\d{6}_[A-Za-z0-9-]+_[A-Za-z12]+\/cell\d+_R\d+)"):
        """Get the cell pseudotime and protein type from the image path
        Input:
        features: a pd dataframe from cell_features.txt
        img_path: path to the .tif file
        regex: regex to match the path column in cell_features.txt
        Output:
        labels = (cell_stage, poi) 
        """
        # match path exp/cell to data/Data_tifs/exp/cell/rawtif/TR1_W0001_P0007_T0024.tif
        #regex = "(\d{6}_[A-Za-z0-9-]+_[A-Za-z12]+\/cell\d+_R\d+)"
        
        match = re.search(regex, img_path).group(0) # will find the path that is in cell_features.txt
        index_time = int(re.search('(?<=00)([0-4]\d)(?=.tif)', img_path).group(0))
    
        match = match.replace('/', '\\') # replace / with \ to match cell_features format
    
        rows = self.cell_features[self.cell_features['path'] == match]['index'].astype(np.uint16)
        row_ind = rows[rows == index_time].index._data[0].astype(np.uint16) # ugly solution? maybe fix later
        
        return self.cell_features.at[row_ind, 'time_2'], self.cell_features.at[row_ind, 'poi']

    def _extract_labels(self, img_path):
        """For extract labels from cell_features from img_path"""
        stage, poi = self._get_cell_stage_from_path(img_path, regex=self.cell_regex)
        poi = protein_dict[poi] # turn str to int
        return stage, poi
    
    
    def preprocess_image(self, image):
        """Convert image to [-1, 1] floats and optionally perform preprocessing in this order:
        1. Percentile normalization
        2. Adding noise to the image
        3. Perform random angle augmentation"""
        # assuming image is a np.array with uint16
        image = (image/32767.5 - 1.0).astype(np.float32)

        if self.normalization:
            # normalization using percentiles
            # r= cell matrix, g=poi, b=nucleus
            low_r, high_r = np.percentile(image[:, :, 0], self.percentiles)
            low_g, high_g = np.percentile(image[:, :, 1], self.percentiles)
            low_b, high_b = np.percentile(image[:, :, 2], self.percentiles)

            # Rescale each channel using its respective percentiles
            rescaled_r = np.clip((image[:, :, 0] - low_r) / (high_r - low_r), -1, 1)
            rescaled_g = np.clip((image[:, :, 1] - low_g) / (high_g - low_g), -1, 1)
            rescaled_b = np.clip((image[:, :, 2] - low_b) / (high_b - low_b), -1, 1)

            # Combine the rescaled channels back into an image
            image = np.stack((rescaled_r, rescaled_g, rescaled_b), axis=-1)
            del rescaled_r, rescaled_g, rescaled_b
        
        # add noise to the images
        if self.add_noise:
            if self.noise_type == 'gaussian':
                # add noise separately to each channel
                #if 'separate channels' in self.kwargs.keys() and self.kwargs['separate channels'] is True:
                mean = 0
                std = np.std(image, axis=(0, 1)) # use the standard deviation of the channel to add noise
                noise = np.zeros_like(image)
                for i in range(len(std)):   
                    noise[:, :, i] = np.random.normal(0, std[i], image[:, :, i].shape)
            else:
                raise NotImplementedError
            image = np.clip(image+noise, -1, 1)
            del noise


        if self.random_angle:
            angle = 360 * random.random()  # random float angle in degrees        
            image = rotate(image, angle, mode=self.rotation_mode)  # rotate image
        
        return image
            


    def __len__(self):
        if self.skip_top_bottom > 0:
            return len(self.img_paths) * (self.z_stacks - 2 * self.skip_top_bottom)
        else:
            return len(self.img_paths) * self.z_stacks
    

    def __getitem__(self, idx):
        image_idx = idx // self.z_stacks if self.skip_top_bottom == 0 else idx // (self.z_stacks - 2 * self.skip_top_bottom)
        stack_idx = idx % self.z_stacks if self.skip_top_bottom == 0 else idx % (self.z_stacks - 2 * self.skip_top_bottom) + self.skip_top_bottom
        img_path = self.img_paths[image_idx]
        image = AICSImage(img_path, reader=TiffReader)
        
        #workaround error with get_image_dask_data
        poi_img = image.get_image_data('XY', C=0, T=0, Z=stack_idx)
        nuc = image.get_image_data('XY', C=0, T=0, Z=stack_idx + self.z_stacks)
        cell = image.get_image_data('XY', C=0, T=0, Z=stack_idx + 2 *self.z_stacks)
        image = np.stack((cell, poi_img, nuc), axis=2)


        # get labels 0: cell stage, 1: stack_idx, 2-33 poi according to protein_dict
        stage, poi = self._extract_labels(img_path) # integers now
        labels = np.array([stage-1, stack_idx, poi]) # now all labels starts with zero to mark the first
        # one hot encoding + rescaling to 0-1 for labels
        #stage = stage/20 # rescale from 1 - 20 to ~0 and 1
        #stack_idx = stack_idx/(self.z_stacks-1) # rescale from 0 to 30 to 0 and 1
        #poi = poi/len(protein_dict.keys()) # rescale from 0 to 31 to 0 and 1
        #labels = np.zeros((2 + len(protein_dict), 1), dtype=np.float32)
        #labels[0], labels[1] = stage, stack_idx
        #labels[poi+2] = 1 # one hot encoding  
        del poi_img, nuc, cell

        if self.return_masks:
            mask_path = img_path.replace('rawtif', 'masktif')
            mask = AICSImage(mask_path, reader=TiffReader)
            cell = mask.get_image_data('XY', C=0, T=0, Z=stack_idx + self.z_stacks) # has zeros and ones
            #nuclei = (mask.get_image_data('XY', C=0, T=0, Z=stack_idx)/2) # has zeros and twos, only use cell volume as reference
            nuclei = np.zeros_like(cell)
            poi_channel = np.zeros_like(cell)
            mask = np.stack((cell, poi_channel, nuclei), axis=2).astype(np.float32) # same channel order as raw images
            del cell, nuclei, poi_channel
            #take image from 0 to 1 to -1, 1 as rawimages
            mask = 2 * mask - 1
            return {'image':self.preprocess_image(image), 'labels': labels, 'mask': mask}
        else:
            return {'image':self.preprocess_image(image), 'labels': labels} 
        
  

class MCAConditionEmbedder(nn.Module):
    
    def __init__(self, output_size_context_dim=128, hidden_layers = [128], one_hot_all_labels = False, concat_mode=False, image_embedding_model="", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        MCA Label Embedder for the LatentDiffusion Model
        Input: 
        output_size_context_dim: the size of the sequential network output. Must match the context_dim of the LatentDiffusion Model
        hidden_layers: sizes of the hidden layers in the label embedding model. ReLu is used as activation function between layers
        one_hot_all_labels: if True one hot encode all labels, if False one hot encode only the protein label (relevant for label embedding)
        concat_mode: if True concatenate the image embedding with the raw images in latent space
        image_embedding_model: if concat_mode is True, pass the image embedding model to downsample the mask for latent space to use with LDM hybrid conditioning mode

        """

        self.one_hot_all_labels = one_hot_all_labels
        self.concat_mode = concat_mode
        
        # load the pretrainer autoencoder as mask embedding model
        if image_embedding_model and self.concat_mode:
            assert not isinstance(image_embedding_model, dict)
            self.image_embedding_model = instantiate_from_config(image_embedding_model)
            self.image_embedding_model_exists = True
        else:
            self.image_embedding_model_exists = False
        
        # prepare the label embedding model
        layers = []
        if one_hot_all_labels:
            in_features = np.sum(list(size_dict.values()))
        else:
            in_features = 2 + len(protein_dict) # only one hot encode the protein label
        for hidden_layer_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_layer_size))
            layers.append(nn.ReLU())
            in_features = hidden_layer_size
        layers.append(nn.Linear(in_features, output_size_context_dim))
        self.label_embedder = nn.Sequential(*layers)
        
       
        
    def forward(self, batch):
        """
        Forward pass for the label embedding model and the image embedding model if concat_mode is True
        """
        if type(batch) == dict and "labels" in batch.keys(): 
            labels = batch["labels"]
        else:
            labels = batch
        if self.one_hot_all_labels:
            one_hot = []
            for i in range(labels.shape[1]):
                one_hot.append(self.one_hot(labels[:, i], i))
            labels = torch.cat(one_hot, 1).to(labels.device)
            del one_hot
        
        else:
            new_labels = torch.zeros(labels.shape[0], len(protein_dict.keys())+ 2)
            new_labels[:, labels[:, 2]+ 2] = 1
            new_labels[:, 0] = labels[:, 0]/size_dict[0] # rescale from 0 to 19 to [0, 1)
            new_labels[:, 1] = labels[:, 1]/size_dict[1] # rescale from 0 to 30 to [0,1)
            labels = new_labels.to(labels.device)
            del new_labels

        # label embedding
        label_embedding = self.label_embedder.to(labels.device)
        cond = {'c_crossattn': [label_embedding(labels)]} 

        # image embedding
        if self.concat_mode and type(batch) == dict and 'mask' in batch.keys() and self.image_embedding_model_exists:
            batch['mask'] = torch.permute(batch['mask'], (0, 3, 1, 2))
            cond['c_concat'] = [ self.image_embedding_model.encode(batch['mask']) ]

        return cond
    

    def one_hot(self, label, label_type):
        """One hot encode the label"""
        # stage 0:19, stack_idx 0:30, poi 0:31
        one_hot = torch.zeros(label.shape[0], size_dict[label_type]).to(label.device)
        for i in range(label.shape[0]):
            one_hot[i, label[i]] = 1
        return one_hot



def test_dataloader(dataset):
        init_fn = None
        dl = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=False, worker_init_fn=init_fn)
        #return DataLoader(self.datasets["train"], batch_size=self.batch_size,
        #                  num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
        #                  worker_init_fn=init_fn, persistent_workers=self.num_workers > 0,pin_memory=True)

        print()


def create_cached_paths():
    dirs = "/proj/aicell/data/stable-diffusion/mca/ftp.ebi.ac.uk/pub/databases/IDR/idr0052-walther-condensinmap/20181113-ftp/MitoSys /proj/aicell/data/stable-diffusion/mca/mitotic_cell_atlas_v1.0.1_fulldata/Data_tifs"
    image_path='*/*/rawtif/*.tif'

    seed=23
    dirs = dirs.split()
    img_paths = []
    # get paths to all images
    for d in dirs:  
        img_paths.extend(sorted(glob.glob(os.path.join(d, image_path)))) # sort to assure reproducibility no matter the os as glob returns paths in arbitrary order
        random.Random(seed).shuffle(img_paths)
    
    print(len(img_paths))

    # create cache file to create test and train set for autoencoder
    train = 'train-90-data-mca.json'
    test = 'test-10-data-mca.json'

    #filename = 'all-data-mca.json'

    split = 0.9 # save 10 % as test data

    split = int(len(img_paths)*split)
    train_img = img_paths[:split]
    test_img = img_paths[split:]

    with open(train, 'w') as file:
        json.dump(train_img, file)

    with open(test, 'w') as file:
        json.dump(test_img, file)


def find_images_with_same_condition(protein_str, mst, z, save_paths=False, only_one_image=False):
    cf_path = "/proj/aicell/data/stable-diffusion/mca/cell_features_necessary_columns.txt"
    dirs = "/proj/aicell/data/stable-diffusion/mca/ftp.ebi.ac.uk/pub/databases/IDR/idr0052-walther-condensinmap/20181113-ftp/MitoSys /proj/aicell/data/stable-diffusion/mca/mitotic_cell_atlas_v1.0.1_fulldata/Data_tifs"
    # Load all data
    dataset = MCACombineDataset(dirs, cf_path, use_cached_paths=True, cached_paths="/proj/aicell/data/stable-diffusion/mca/all-data-mca.json",
                                 group='train', train_val_split=1.0, normalization=True)
    # check if the input is correct
    
    assert protein_str in protein_dict.keys() 
    assert mst in range(1,21), 'mst must be between 1 and 20 as that is how it is defined in the cell_features.txt file but later the mst label will be rescaled to 0-19'
    assert z in range(31) 

    
    # remove the image paths that do not have protein
    # faster to loop through the paths than the images
    i=0  
    protein = protein_dict[protein_str] 
    
    while i <  len(dataset.img_paths):
        img_path = dataset.img_paths[i]
        stage, poi = dataset._extract_labels(img_path)
        if poi == protein and stage == mst:
            i+=1
            if only_one_image:
                # the first path in the list has the same protein and mst
                # this means the first 0-30 idx images in the dataset has the same protein and mst
                # now only choose the required z-stack
                assert dataset[z]['labels'][0] == mst-1
                assert dataset[z]['labels'][2] == protein
                assert dataset[z]['labels'][1] == z
                return [dataset[z]]
                
        else:
            dataset.img_paths.remove(img_path)
    
    #save the paths to images with the same protein and mst
    if save_paths:
        with open(f'all_paths_mst_{mst}_poi_{protein_str}.json', 'w') as file:
            json.dump(dataset.img_paths, file)
    
    
    # find images with the same z-stack position
    # now the only images left should be the ones with the same protein and mst
    images_with_same_z = []
    
    for i in range(0, len(dataset)):
        if i % dataset.z_stacks == z:
            images_with_same_z.append(dataset[i])

            assert dataset[i]['labels'][0] == mst-1
            assert dataset[i]['labels'][2] == protein
            assert dataset[i]['labels'][1] == z
            
    assert len(images_with_same_z) == len(dataset.img_paths)
    
    return images_with_same_z


def main():
    find_images_with_same_condition(protein_str='STAG2', mst=6, z=5, save_paths=True)

    exit()
    dirs = "/proj/aicell/data/stable-diffusion/mca/ftp.ebi.ac.uk/pub/databases/IDR/idr0052-walther-condensinmap/20181113-ftp/MitoSys /proj/aicell/data/stable-diffusion/mca/mitotic_cell_atlas_v1.0.1_fulldata/Data_tifs"
    cf_path = "/proj/aicell/data/stable-diffusion/mca/cell_features_necessary_columns.txt"

    dirs = "/data/ethyni/mca/mitotic_cell_atlas_v1.0.1_exampledata/Data_tifs"
    cf_path="/data/ethyni/mca/cell_features_necessary_columns.txt"
    train = MCACombineDataset(dirs, cf_path, group='train', z_stacks=31,
                              add_noise=False, normalization=True, random_angle=True, image_path='*/*/rawtif/*.tif', 
                              train_val_split=1, use_cached_paths=False, return_masks=True, skip_top_bottom=3)
    print(len(train))
    for i in range(len(train)):
        train[i]
    # create cache files
    #create_cached_paths()
    exit()

    # test time if cache is faster or not
    cache= '/proj/aicell/users/x_emmku/stable-diffusion/ldm/data/all-data-mca.json'
    ts_cache= []

    ts_no_cache = []

    for i in range(1000):
        train = MCACombineDataset(dirs, cf_path, use_cached_paths=False, group='train')
        ts_no_cache.append(train.init_time)
        train2 = MCACombineDataset(dirs, cf_path, use_cached_paths=True, 
        cached_paths=cache, group='train')
        ts_cache.append(train2.init_time)
    
    print(np.mean(ts_cache), 'mean init time cached')
    print(np.mean(ts_no_cache), 'mean init no cache')
        

    





if __name__=='__main__':
    main()