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

# all available proteins from v1.0.1 and v1.1
protein_dict = {'APC2': 0, 'AURKB': 1, 'BUB1': 2, 'BUBR1': 3, 
                'CDCA8': 4, 'CENPA': 5, 'CEP192': 6, 'CEP250': 7, 
                'CTCF': 8, 'KIF11': 9, 'KIF4A': 10, 'MAD2L1': 11, 
                'MIS12': 12, 'NCAPH2': 13, 'NEDD1': 14, 'NES': 15, 
                'NUP107': 16, 'NUP214': 17, 'PLK1': 18, 'RACGAP1': 19,
                'RANBP2': 20, 'SCC1': 21, 'STAG1': 22, 'STAG2': 23, 
                'TOP2A': 24, 'TPR': 25, 'TUBB2C': 26, 'WAPL': 27,
                'NCAPH': 28, 'NCAPD2': 29, 'NCAPD3': 30, 'SMC4': 31 } 

        
class MCACombineDataset(Dataset):
    """Dataset class for the mitotic cell atlas raw images"""
    def __init__(self, directories, cell_features, use_cached_paths=True, cached_paths='', group='train', z_stacks=31, seed=123, train_val_split=0.95, image_path='*/*/rawtif/*.tif',
                 random_angle=False, rotation_mode='reflect', normalization=False, percentiles=(1, 99), add_noise=False,
                 noise_type = 'gaussian'):
        """Input:
        directories: paths to all relevant directories with the experiment folders. Multiple paths should be separated by any whitespace character
        cell_features: path to cell_features_necessary_columns.txt
        seed and train_val_split to split the dataset into train and val
        image_paths: relative glob style path from directories path to the .tif files
        
        Extra things:
        random_angle: if True apply random angle transformation to images
        rotation_mode: mode for scimage.transform.rotate if random_angle
        normalization: if True apply percentile normalization to images for each channel separately
        percentiles: tuple with percentiles to use for normalization if normalization
        add_noise: if True add noise to the images
        noise_type: str with the noise type. Currently Gaussian noise is the only implemented type which adds gaussian with zero mean and std(channel) separately to the images
        data_to_use: float for deciding how large dataset you want to use. Used for development purposes to reduce dataset size easily
        whilst still accessing all the data"""
        #t=time.time()
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
        self.cell_features = pd.read_csv(cell_features, sep='\s+')
        self.z_stacks = z_stacks
        self.random_angle = random_angle
        self.rotation_mode = rotation_mode if self.random_angle else None
        self.normalization = normalization
        self.percentiles = percentiles if self.normalization else None
        self.add_noise = add_noise
        self.noise_type = noise_type if self.add_noise else None
        #self.init_time = time.time() - t

    
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
        stage, poi = self._get_cell_stage_from_path(img_path)
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
        return len(self.img_paths) * self.z_stacks
    

    def __getitem__(self, idx):
        image_idx = idx // self.z_stacks
        stack_idx = idx % self.z_stacks
        img_path = self.img_paths[image_idx]
        image = AICSImage(img_path, reader=TiffReader)
        #image = image.get_image_dask_data('ZYX')
        # poi channel, nuc channel, cell channel is original order
        #change order here already to prepare for rgb format, r= cell, g=poi, b=nuc
        #image = image[[stack_idx + 2 * self.z_stacks, stack_idx, stack_idx + self.z_stacks], :, :]

        #workaround error with get_image_dask_data
        poi_img = image.get_image_data('XY', C=0, T=0, Z=stack_idx)
        nuc = image.get_image_data('XY', C=0, T=0, Z=stack_idx + self.z_stacks)
        cell = image.get_image_data('XY', C=0, T=0, Z=stack_idx + 2 *self.z_stacks)
        image = np.stack((cell, poi_img, nuc), axis=2)


        # get labels 0: cell stage, 1: stack_idx, 2-30 poi according to protein_dict
        stage, poi = self._extract_labels(img_path) # integers now
        #labels = (stage, stack_idx, poi)
        # one hot encoding + rescaling to 0-1 for labels
        stage = stage/20 # rescale from 1 - 20 to ~0 and 1
        stack_idx = stack_idx/(self.z_stacks-1) # rescale from 0 to 30 to 0 and 1
        labels = np.zeros((2 + len(protein_dict), 1), dtype=np.float32)
        labels[0], labels[1] = stage, stack_idx
        labels[poi+2] = 1 # one hot encoding  
        
        return {'image':self.preprocess_image(image), 'labels': labels} 
        
    
    # if it is easier with the generator, then could rename current getitem to a function func and uncomment this
    #def _sample_generator(self):
    #    for i in range(len(self)):
    #        yield func[i]
    
    #def __getitem__(self):
    #    return next(self.sample_generator)



class MCALabelEmbedder(nn.Module):
    
    def __init__(self, layer_1_size, output_size_context_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.label_embedder = nn.Sequential(
                nn.Linear(len(protein_dict.keys()) + 2, layer_1_size),
                nn.ReLU(),
                nn.Linear(layer_1_size, output_size_context_dim),
            )
    

    def forward(self, batch, key='labels'):
        if type(batch) == dict: # only use the labels for the conditioning in image logger
            batch = batch[key]
        embedding = self.label_embedder.to(batch.device)
        if batch.shape[-1] == 1:
            batch = torch.reshape(batch, (batch.shape[0], batch.shape[1]))
        return embedding(batch)



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
    

def main():
    
    dirs = "/proj/aicell/data/stable-diffusion/mca/ftp.ebi.ac.uk/pub/databases/IDR/idr0052-walther-condensinmap/20181113-ftp/MitoSys /proj/aicell/data/stable-diffusion/mca/mitotic_cell_atlas_v1.0.1_fulldata/Data_tifs"
    cf_path = "/proj/aicell/data/stable-diffusion/mca/cell_features_necessary_columns.txt"
    #train = MCACombineDataset(dirs, cf_path, 'train', z_stacks=31,
    #                          add_noise=False, normalization=True, random_angle=True, image_path='*/*/rawtif/*.tif', 
    #                          train_val_split=1)
    
    # create cache files
    create_cached_paths()
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