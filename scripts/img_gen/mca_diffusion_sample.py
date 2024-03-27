import argparse, os, datetime
from collections import defaultdict
import random
import colorsys
import pickle

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch, torchvision
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.parse import str2bool
from ldm.util import instantiate_from_config
from ldm.evaluation.metrics import calc_metrics
# import yaml
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

from ldm.data.mca import protein_dict
from torchvision.utils import make_grid

"""
Command example: CUDA_VISIBLE_DEVICES=0 python scripts/prot2img.py --config=configs/latent-diffusion/hpa-ldm-vq-4-hybrid-protein-location-augmentation.yaml --checkpoint=logs/2023-04-07T01-25-41_hpa-ldm-vq-4-hybrid-protein-location-augmentation/checkpoints/last.ckpt --scale=2 --outdir=./data/22-fixed --fix-reference

"""

def condtions_to_text(c):
    # stage, stack_idx, poi
    c = c.cpu()
    stage = int(c[:, 0]*20)
    stack_idx = int(c[:, 1]*31)
    poi = torch.argmax(c[:, 2:])
    poi = list(protein_dict.values()).index(poi)
    poi = list(protein_dict.keys())[poi]

    return stage, stack_idx, poi



def main(opt):
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    split = "test"
    if opt.name:
        name = opt.name
    else:
        name = f"{split}__gd{opt.scale}__steps{opt.steps}"
    # nowname = now + "_" + name
    opt.outdir = f"{os.path.dirname(os.path.dirname(opt.checkpoint))}/{name}"

    config = OmegaConf.load(opt.config)
    data_config = config['data']
    
    # Load data
    data = instantiate_from_config(data_config)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    
    # Load the model checkpoint
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.checkpoint, map_location="cpu")["state_dict"],
                          strict=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)

    total_count = len(data.datasets[split])
     
    predicted_images, gt_images = [], []
    filenames, conditions = [], []
    mse_list, ssim_list = [], []
    with torch.no_grad():
        with model.ema_scope():
            for i, sample in tqdm(enumerate(data.datasets[split]), total=total_count):
                if i % opt.skip_images != 0:
                    continue
                sample = {k: torch.from_numpy(np.expand_dims(sample[k], axis=0)).to(device) if isinstance(sample[k], (np.ndarray, np.generic)) else sample[k] for k in sample.keys()}
                name = f"{k}_image{i}"
                outpath = os.path.join(opt.outdir, name)

                c = model.cond_stage_model(sample)
                uc = {'c_concat': [torch.zeros_like(c)], 'c_crossattn': [c]}
                sample['image'] = sample['image'].permute(0, 3, 1, 2)
                z = model.encode_first_stage(sample['image'])
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=z.shape[0],
                                                 shape=z.shape[1:],
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                mse, ssim = calc_metrics(x_samples_ddim, sample['image'])
                mse_list.append(mse)
                ssim_list.append(ssim)

                #gt_image = torch.clamp((sample['image']+1.0)/2.0, min=0.0, max=1.0)
                #predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                
                gt_image = sample['image'].squeeze(0)
                predicted_image = x_samples_ddim.squeeze(0)

                stage, stack_idx, poi = condtions_to_text(sample['labels'])
                conditions.append(f"Cell stage:{stage} stack position:{stack_idx} and POI:{poi}")
                predicted_images.append(predicted_image)
                gt_images.append(gt_image)
                filenames.append(name)

    n_images_to_plot = min(8, len(predicted_images))
    fig, axes = plt.subplots(n_images_to_plot, 2, figsize=(10,10))   
    if opt.pil:
        # pred_captions = [f"Predicted, MSE: {mse_list[i]:.2g}, SSIM: {ssim_list[i]:.2g}" for i in range(len(predicted_images))]
        T = torchvision.transforms.ToPILImage()
        for i in range(n_images_to_plot):
            for j in range(2):
                ax = axes[i, j]
                if j == 0:
                    image = T(predicted_images[i]) 
                    title = f"Predicted, MSE: {mse_list[i]:.2g}, SSIM: {ssim_list[i]:.2g}"
                else:
                    image = T(gt_images[i])
                    title = f"GT, {conditions[i]}"
                ax.imshow(image)
                ax.axis('off')
                ax.set_title(title)
    
    if not opt.pil:
        for i in range(n_images_to_plot):
            for j in range(2):
                ax = axes[i, j]
                # Use mean instead of sum, which was the previous practice
                if j == 0:
                    image = predicted_images[i] #.mean(axis=2)
                    title = f"Predicted, MSE: {mse_list[i]:.2g}, SSIM: {ssim_list[i]:.2g}"
                else:
                    image = gt_images[i] #.mean(axis=2)
                    title = f"GT, {conditions[i]}"
                print(f"max: {image.max()}, min:{image.min()}")
                image = image.cpu().numpy().transpose(0,2,3,1)[0]*255
                image = image.astype(np.uint8)
                # clip the image to 0-1
                image = np.clip(image, 0, 255) / 255.0
                ax.imshow(image, vmin=0, vmax=255)
                ax.axis('off')
                ax.set_title(title)
    mse_mean = np.mean(mse_list)
    ssim_mean = np.mean(ssim_list) 
    fig.suptitle(f'{split} reference, guidance scale={opt.scale}, DDIM steps={opt.steps}, MSE: {mse_mean:.2g}, SSIM: {ssim_mean:.2g}')
    fig.savefig(os.path.join(opt.outdir, f'predicted-image-grid-s{opt.scale}.png'))
    fig.tight_layout()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict protein images. Example command: python scripts/prot2img.py --config=configs/latent-diffusion/hpa-ldm-vq-4-hybrid-protein-location-augmentation.yaml --checkpoint=logs/2023-04-07T01-25-41_hpa-ldm-vq-4-hybrid-protein-location-augmentation/checkpoints/last.ckpt --scale=2 --outdir=./data/22-fixed --fix-reference")
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="the model config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        help="the model checkpoint",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="?",
        default="1",
        help="set os.environ['CUDA_VISIBLE_DEVICES']",
    )
    parser.add_argument(
        "--pil",
        type=str2bool,
        nargs="?",
        default=True,
        help="if true use pil to save images, otherwise use matplotlib",
    )
    parser.add_argument(
        "--skip_images",
        type=int,
        nargs="?",
        default=1,
        help="how much of the dataset to generate images from",
    )
      
    opt = parser.parse_args()

    main(opt)
