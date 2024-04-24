from aicsimageio.aics_image import AICSImage
from aicsimageio.readers import TiffReader
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rotate
import random
import re
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torchvision.transforms import ToPILImage
from ldm.data.mca import protein_dict

def normalize_image_percentile(image, channel) -> None:
    # normalization using percentiles  
    percentiles = torch.FloatTensor([0.01, 0.99]).to(image.device) 
    low, high = torch.quantile(image[channel, :, :], percentiles)
    image[channel, :, :] = torch.clamp((image[channel, :, :] - low) / (high - low), -1, 1)
    


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
    random_indices = np.random.choice(len(data.datasets['train']), num_images)

    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            for i in tqdm(random_indices, desc='Processing images'):
                x = data.datasets['train'][i]['image']
                position = int_list.index(data.datasets['train'][i]['labels'][-1])
                poi = poi_list[position]
                x = torch.from_numpy(x)
                x = torch.permute(x, (2, 0, 1))
                x_input = x.unsqueeze(0) 
                xhat, _ = model(x_input)
                xhat = torch.squeeze(xhat.detach())
                
                # normalize reconstruction in the same way that the ground truth MCA images are normalized
                for j in range(3):
                    normalize_image_percentile(xhat, channel=j)
                
                xhat = T(xhat)
                x = T(x)

                # save images in logdir
                if save_input:
                    x.save(os.path.join(logdir, f"input__{i}_poi{poi}.png"))
                if save_recs:
                    xhat.save(os.path.join(logdir, f"reconstruction_{i}_poi{poi}.png"))
        

# Function to load images from folder
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename))
            images.append(img)
    return images

# Function to create a grid of images
def create_image_grid(images, cols, rows):
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    grid_image = Image.new('RGB', (width * cols, height * rows))
    for i, img in enumerate(images):
        col = i % cols
        row = i // cols
        grid_image.paste(img, (col * width, row * height))
    return grid_image

# Function to add header
def add_header(image, text, font_size=20):
    width, _ = image.size
    font = ImageFont.load_default()
    font = font.font_variant(size=font_size)
    draw = ImageDraw.Draw(image)
    text_width, _ = draw.textsize(text, font=font)
    draw.text(((width - text_width) // 2, 10), text, fill=(255, 255, 255), font=font)
    return image

def create_comparison_image(input_folder_path, reconstruction_folder_path):

    # Load images
    input_images = load_images(input_folder_path)
    reconstruction_images = load_images(reconstruction_folder_path)

    # Sort images based on filenames for comparison
    input_images.sort(key=lambda x: x.filename)
    reconstruction_images.sort(key=lambda x: x.filename)

    # Create grids of images for each column
    cols = 6
    rows = 6

    num_images = int(rows * cols / 2)
    random_indices = np.random.choice(len(input_images), num_images)

    #print(random_indices)
    #rands=[9, 37, 96, 83, 198, 194, 94, 19, 141, 62 ,120, 48, 57, 21, 105 , 182, 165, 90]
    use_input = []
    use_reconstruction = []
    #for j in rands:
    for j in random_indices:
        use_input.append(input_images[j])
        use_reconstruction.append(reconstruction_images[j])

    input_grids = [create_image_grid(use_input[i::3], 1, rows) for i in range(3)]
    reconstruction_grids = [create_image_grid(use_reconstruction[i::3], 1, rows) for i in range(3)]

    # Add headers to each grid
    #input_grids_with_headers = [add_header(grid, f"Input {i+1}") for i, grid in enumerate(input_grids)]
    #reconstruction_grids_with_headers = [add_header(grid, f"Reconstruction {i+1}") for i, grid in enumerate(reconstruction_grids)]

    # Combine grids side by side
    combined_width = sum(grid.width for grid in input_grids + reconstruction_grids)
    combined_height = max(grid.height for grid in input_grids + reconstruction_grids)
    combined_grid = Image.new('RGB', (combined_width, combined_height))
    x_offset = 0
    for input_grid, reconstruction_grid in zip(input_grids, reconstruction_grids):
        combined_grid.paste(input_grid, (x_offset, 0))
        combined_grid.paste(reconstruction_grid, (x_offset + input_grid.width, 0))
        x_offset += input_grid.width + reconstruction_grid.width

    # Save the combined grid image
    combined_grid.save("comparison_grid_alternating_v2.png", 'PNG')




def main():
    cfg = "/proj/aicell/users/x_emmku/stable-diffusion/configs/autoencoder/sample_autoencoder_images_mca.yaml"
    logdir = "/proj/aicell/data/stable-diffusion/mca/images_for_report/autoencoder_gt_and_rec"
    #generate_autoencoder_input_and_reconstruction(cfg, logdir, num_images=200)

    recs = "/proj/aicell/data/stable-diffusion/mca/images_for_report/autoencoder_gt_and_rec/recs"
    inputs = "/proj/aicell/data/stable-diffusion/mca/images_for_report/autoencoder_gt_and_rec/input"

    create_comparison_image(inputs, recs)



if __name__ == '__main__':
    main()
