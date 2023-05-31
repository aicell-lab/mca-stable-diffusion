

import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import yaml
import matplotlib.pyplot as plt
import imageio


# Read images from the data directory
# The images are saved by the following code
# ref_image = ref_image.cpu().numpy()[0]*255
# Image.fromarray(ref_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_reference.png')

# predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
# Image.fromarray(predicted_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_prediction.png')

# predicted_images.append(predicted_image)
# locations.append(sample['info']['locations'])

data_dir = "data/2-densenet"

fixed_ref = "6_G1_2_Nuclear speckles_reference.png"
# list all the files in the folder
fixed_ref = np.array(Image.open(os.path.join(data_dir, fixed_ref)))

# create a tif file for storing the processed images using imageio

imgs = []
for file in os.listdir(data_dir):
    if file.endswith("_prediction.png"):
        # THe file names are like this: 10_G5_1_Cytosol_prediction.png
        file_path = os.path.join(data_dir, file)
        # Take the first channel of the prediction image then replace the green channel in the fixed_ref image
        fixed = fixed_ref.copy()
        pred = np.array(Image.open(file_path))[:, :, 0]
        fixed[:, :, 1] = pred
        # draw the text on the image with the file name
        merged = Image.fromarray(fixed)
        draw = ImageDraw.Draw(merged)
        name = file.rstrip('_prediction.png')
        name = "_".join(name.split("_")[3:])
        draw.text((0, 0), name, (255, 255, 255))
        merged = np.array(merged)
        imageio.imwrite(file_path.replace("_prediction.png", "_merged.png"), merged)
        imgs.append(merged)

imageio.mimwrite(f"{data_dir}/stack.tif",imgs)

# create a gif animation from the tif file
imageio.mimsave(f"{data_dir}/stack.gif",imgs, duration=1)
        
        