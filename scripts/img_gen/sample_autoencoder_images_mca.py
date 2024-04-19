from aicsimageio.aics_image import AICSImage
from aicsimageio.readers import TiffReader
import numpy as np
import cv2
import os
from PIL import Image
from skimage.transform import rotate
import random
import re
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torchvision.transforms import ToPILImage
from ldm.data.mca import protein_dict


def generate_autoencoder_input_and_reconstruction(cfg, logdir, num_images=10, save_input=True, save_recs=True):
    
    config = OmegaConf.load(cfg)
     
    # model
    model = instantiate_from_config(config.model)

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        
    poi_list = list(protein_dict.keys())
    int_list = list(protein_dict.values())

    T = ToPILImage()
    random_incides = np.random.choice(len(data.datasets['train']), num_images)

    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            for i in tqdm(random_incides, desc='Processing images'):
                x = data.datasets['train'][i]['image']
                position = int_list.index(data.datasets['train'][i]['labels'][-1])
                poi = poi_list[position]
                x = torch.from_numpy(x)
                x = torch.permute(x, (2, 0, 1))
                x_input = x.unsqueeze(0) 
                xhat, _ = model(x_input)
                xhat.detach()
                xhat = torch.squeeze(xhat)
                xhat = torch.clamp(xhat, -1., 1.)
                xhat = T(xhat)
                x = T(x)

                # save images in logdir
                if save_input:
                    x.save(os.path.join(logdir, f"input__{i}_poi{poi}.png"))
                if save_recs:
                    xhat.save(os.path.join(logdir, f"output_{i}_poi{poi}.png"))
        



def main():
    cfg = "/proj/aicell/users/x_emmku/stable-diffusion/configs/autoencoder/sample_autoencoder_images_mca.yaml"
    logdir = "/proj/aicell/users/x_emmku/stable-diffusion/report/autoencoder_images"
    generate_autoencoder_input_and_reconstruction(cfg, logdir, num_images=10)



if __name__ == '__main__':
    main()
