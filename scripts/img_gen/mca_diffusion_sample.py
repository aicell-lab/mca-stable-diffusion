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
from ldm.data.mca import *

from ldm.data.mca import protein_dict
from torchvision.utils import make_grid

"""
Command example: CUDA_VISIBLE_DEVICES=0 python scripts/prot2img.py --config=configs/latent-diffusion/hpa-ldm-vq-4-hybrid-protein-location-augmentation.yaml --checkpoint=logs/2023-04-07T01-25-41_hpa-ldm-vq-4-hybrid-protein-location-augmentation/checkpoints/last.ckpt --scale=2 --outdir=./data/22-fixed --fix-reference

"""
T = torchvision.transforms.ToPILImage() 

def condtions_to_text(c):
    """Convert the labels to text"""
    # stage, stack_idx, poi
    c = c.cpu()
    stage = int(c[:, 0])
    stack_idx = int(c[:, 1])
    #poi = torch.argmax(c[:, 2:])
    poi = c[:, 2]
    poi = list(protein_dict.values()).index(poi)
    poi = list(protein_dict.keys())[poi]

    return stage, stack_idx, poi


def normalize_image_percentile(image, channel, percentiles=torch.FloatTensor([0.01, 0.99])) -> None:
    """Renormalize the generated images in the same way that the ground truth images were normalized"""
    # normalization using percentiles   
    percentiles = torch.FloatTensor([0.01, 0.99]).to(image.device)
    low, high = torch.quantile(image[channel, :, :], percentiles)
    image[channel, :, :] = torch.clamp((image[channel, :, :] - low) / (high - low), -1, 1)


def generate_images_by_cond(now, opt, model, data, device, sampler, split="test"):
    """
    Generate any number of images with a certain condition decided by the user
    """
    keys = ['mst', 'z', 'protein_str', 'num_images']
    for key in keys:
        assert key in opt.kwargs.keys(), f"Missing key {key} in opt.kwargs"
    mst = int(opt.kwargs['mst'])
    z = int(opt.kwargs['z'])
    protein_str = opt.kwargs['protein_str']
    num_images = int(opt.kwargs['num_images'])

    name = f"mst_{mst}_z_{z}_poi_{protein_str}"

    os.makedirs(opt.outdir, exist_ok=True)
    save_folder = os.path.join(opt.outdir, name)
    os.makedirs(save_folder, exist_ok=True)

    print("Finding images with the same condition...")
    data = find_images_with_same_condition(protein_str=protein_str, mst=mst, z=z, only_one_image=True)
    sample = data[0]
    sample = {k: torch.from_numpy(np.expand_dims(sample[k], axis=0)).to(device) if isinstance(sample[k], (np.ndarray, np.generic)) else sample[k] for k in sample.keys()}
    
    # get conditions from the sample
    model.eval()
    c = model.cond_stage_model(sample)
    uc = {'c_concat': c['c_concat'] if 'c_concat' in c.keys() else torch.zeros_like(c['c_crossattn'][0]), 'c_crossattn': c['c_crossattn']}
    sample['image'] = sample['image'].permute(0, 3, 1, 2)
    z = model.encode_first_stage(sample['image'])
    

    # generate images
    with torch.no_grad():
        with model.ema_scope():
            for i in tqdm(num_images):
                img_name = f"image{i}__{now}"
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                conditioning=c,
                                                batch_size=z.shape[0],
                                                shape=z.shape[1:],
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                predicted_image = x_samples_ddim.squeeze(0)
                for j in range(3):
                        normalize_image_percentile(predicted_image, channel=j)

                img = T(predicted_image)                   
                img.save(os.path.join(save_folder, img_name))
                



def multiple_image_generation(now, opt, model, data, device, sampler,  split="test"):
    """
    Generate many images based on one ground truth image.  The conditions are randomly selected.
    """
    keys=['num_types', 'num_images']
    for k in keys:
        assert k in opt.kwargs.keys(), f"Missing key {k} in opt.kwargs"
    num_images = int(opt.kwargs['num_images'])
    num_types = int(opt.kwargs['num_types'])

    os.makedirs(opt.outdir, exist_ok=True)

    gt_folder = os.path.join(opt.outdir,"ground_truth_images")
    os.makedirs(gt_folder, exist_ok=True)

    condition_folders = []


    total_count = len(data.datasets[split])
    random_indices = random.sample(range(total_count), num_types)

    predicted_images, gt_images = [], []
    filenames, conditions = [], []
    mse_list, ssim_list = [], []

    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            for i in tqdm(random_indices):
                    for n in range(num_images):
                        sample = data.datasets[split][i]
                        sample = {k: torch.from_numpy(np.expand_dims(sample[k], axis=0)).to(device) if isinstance(sample[k], (np.ndarray, np.generic)) else sample[k] for k in sample.keys()}
                        name = f"image{i}_number{n}_{now}"
                        outpath = os.path.join(opt.outdir, name)

                        stage, stack_idx, poi = condtions_to_text(sample['labels'])
                        conditions.append(f"MST_{stage}_z_{stack_idx}_POI_{poi}")
                        if n == 0:
                            save_folder = os.path.join(opt.outdir, conditions[-1])
                            os.makedirs(save_folder, exist_ok=True)
                            condition_folders.append(save_folder)

                        c = model.cond_stage_model(sample)
                        uc = {'c_concat': c['c_concat'] if 'c_concat' in c.keys() else torch.zeros_like(c['c_crossattn'][0]), 'c_crossattn': c['c_crossattn']}
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

                        gt_image = sample['image'].squeeze(0)
                        predicted_image = x_samples_ddim.squeeze(0)

                        for j in range(3):
                                normalize_image_percentile(predicted_image, channel=j)

                        
                        predicted_images.append(predicted_image)
                        gt_images.append(gt_image)
                        filenames.append(name)

                        img = T(predicted_image)                   
                        img.save(os.path.join(save_folder, name + conditions[-1]+"-pred.png"))
                        if n == 0:
                            img = T(gt_image)
                            img.save(os.path.join(gt_folder, name + conditions[-1] +"-gt.png"))


def single_image_generation(now, opt, model, data, device, sampler,  split="test"):
    """
    Genrerate one image per one ground truth image. The ground truth images are all parsed over (unless opt.skip_images>1).
    """
    # go over opt.kwargs
    keys = ['skip_images', 'comparison_image']
    for k in keys:
        assert k in opt.kwargs.keys(), f"Missing key {k} in opt.kwargs"
    skip_images = int(opt.kwargs['skip_images'])
    comparison_image = bool(opt.kwargs['comparison_image'])

    gt_folder = os.path.join(opt.outdir,"ground_truth")
    pred_folder = os.path.join(opt.outdir, "predicted")
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)

    total_count = len(data.datasets[split])
    
    predicted_images, gt_images = [], []
    filenames, conditions = [], []
    mse_list, ssim_list = [], []
    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            for i in tqdm(range(0, total_count, skip_images)):
                sample = data.datasets[split][i]
                sample = {k: torch.from_numpy(np.expand_dims(sample[k], axis=0)).to(device) if isinstance(sample[k], (np.ndarray, np.generic)) else sample[k] for k in sample.keys()}
                name = f"image{i}_{now}"

                c = model.cond_stage_model(sample)
                uc = {'c_concat': c['c_concat'] if 'c_concat' in c.keys() else torch.zeros_like(c['c_crossattn'][0]), 'c_crossattn': c['c_crossattn']}
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

                for j in range(3):
                        normalize_image_percentile(predicted_image, channel=j)
                
                #predicted_image = torch.clamp((predicted_image+1.0)/2.0, min=0.0, max=1.0)

                stage, stack_idx, poi = condtions_to_text(sample['labels'])
                conditions.append(f"Cell stage:{stage}, stack position:{stack_idx}, and POI:{poi}")
                predicted_images.append(predicted_image)
                gt_images.append(gt_image)
                filenames.append(name)

                img = T(predicted_image)                   
                img.save(os.path.join(pred_folder, name + "-pred.png"))
                img = T(gt_image)
                img.save(os.path.join(gt_folder, name + "-gt.png"))


    if comparison_image:
        n_images_to_plot = min(4, len(predicted_images))
        fig, axes = plt.subplots(n_images_to_plot, 2, figsize=(10,10)) 
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
                ax.set_title(title, size=6)
        mse_mean = np.mean(mse_list)
        ssim_mean = np.mean(ssim_list) 
        fig.suptitle(f'{split} reference, guidance scale={opt.scale}, DDIM steps={opt.steps}, MSE: {mse_mean:.2g}, SSIM: {ssim_mean:.2g}')
        fig.savefig(os.path.join(opt.outdir, f'predicted-image-grid-s{opt.scale}-{now}.png'))
        fig.tight_layout()



def save_gt_images_with_same_condition(opt, device):
    
    keys = ['mst', 'z', 'protein_str', 'save_path']
    for key in keys:
        assert key in opt.kwargs.keys(), f"Missing key {key} in opt.kwargs"
    
    mst = int(opt.kwargs['mst'])
    z = int(opt.kwargs['z'])
    assert os.path.exists(opt.kwargs['save_path']), f"Save path {opt.kwargs['save_path']} does not exist"
    dir_name = f"mst_{mst}_z_{z}_poi_{opt.kwargs['protein_str']}"
    outdir = os.path.join(opt.kwargs['save_path'], dir_name)
    os.makedirs(outdir, exist_ok=True)

    print("Finding images with the same condition...")
    data = find_images_with_same_condition(protein_str=opt.kwargs['protein_str'], mst=mst, z=z)

    print("Saving ground truth images with the same condition...")
    total_count = len(data)
    for i in tqdm(range(0, total_count)):
        sample = data[i]
        sample = {k: sample[k] for k in sample.keys()}
        name = f"image{i}"
        img = Image.fromarray(((sample['image']+1)/2*255).astype(np.uint8))
        img.save(os.path.join(outdir, name + ".png"))
    print(f"Saved {total_count} images with the same condition to {outdir}")



class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for value in values:
            key, val = value.split("=")
            kwargs[key] = val
        setattr(namespace, self.dest, kwargs)        
                
                               
def main(opt):
    """
    Main function for generating predicted images using the MCA diffusion model.

    Args:
        opt (argparse.Namespace): Command-line arguments and options.

    Returns:
        None
    """
    if opt.function_to_run == save_gt_images_with_same_condition:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        save_gt_images_with_same_condition(opt, device)
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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
        if opt.checkpoint != '':
            model.load_state_dict(torch.load(opt.checkpoint, map_location="cpu")["state_dict"],strict=False)

        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        print(f"Function to run: {opt.function_to_run}")
        opt.function_to_run(now, opt, model, data, device, sampler,  split="test")
        
            
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mca images. Example command: python scripts/img_gen/mca_diffusion_sample.py --checkpoint=/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round3-2024-04-15T13-03-57_mca_debug/checkpoints/epoch=000002.ckpt --config=/proj/aicell/users/x_emmku/stable-diffusion/configs/latent-diffusion/mca_gen_imgs_debug.yaml --gpu=0 --skip_images=100 --scale=10 --steps=100 --save_images=True" )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="the model config"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        help="the model checkpoint"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="unconditional guidance scale"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="?",
        default="0",
        help="set os.environ['CUDA_VISIBLE_DEVICES']",
        required=False
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
        required=False
    )
    parser.add_argument(
        '-f',
        dest='function_to_run',
        choices=['single', 'multiple', 'conditions', 'gt_images'],
        default='single',
        required=True
    )
    parser.add_argument(
        '-k',
        '--kwargs',
        nargs='*',
        action=ParseKwargs,
        required=True
    )
      
    opt = parser.parse_args()

    function_map = {'single': single_image_generation, 'multiple': multiple_image_generation, 
                    'conditions': generate_images_by_cond, 'gt_images': save_gt_images_with_same_condition}

    opt.function_to_run = function_map[opt.function_to_run]

    main(opt)