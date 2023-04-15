import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import yaml
import matplotlib.pyplot as plt

"""
Command example: python scripts/prot2img.py --config=configs/latent-diffusion/hpa-ldm-vq-4-hybrid-protein-location-augmentation.yaml --checkpoint=logs/2023-04-07T01-25-41_hpa-ldm-vq-4-hybrid-protein-location-augmentation/checkpoints/last.ckpt --scale=2 --outdir=./data/22-fixed --fix-reference

"""

data_config_yaml = """
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 48
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
      params:
        seed: 123
        train_split_ratio: 0.95
        group: 'train'
        cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-1000.pickle
        channels: [1, 1, 1]
        return_info: true
        filter_func: has_location
        include_location: true
    validation:
      target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
      params:
        seed: 123
        train_split_ratio: 0.95
        group: 'validation'
        cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-1000.pickle
        channels: [1, 1, 1]
        return_info: true
        filter_func: has_location
        include_location: true
"""


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


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
        "--fix-reference",
        action="store_true",
        help="fix the reference channel",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()


    config = yaml.safe_load(data_config_yaml)
    data_config = config['data']

    # data
    data = instantiate_from_config(data_config)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # each image is:
    # 'image': array(...)
    # 'file_path_': 'data/celeba/data256x256/21508.jpg'
    

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.checkpoint)["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    ref = None
    os.makedirs(opt.outdir, exist_ok=True)
    total_count = len(data.datasets['validation'])
    predicted_images = []
    locations = []
    with torch.no_grad():
        with model.ema_scope():
            for sample in tqdm(data.datasets['validation']):
            # for image, mask in tqdm(zip(images, masks)):
                # print(d['info']['Ab state'], d['info']['locations'], d['location_classes'])
                sample = {k: torch.from_numpy(np.expand_dims(sample[k], axis=0)).to(device) if isinstance(sample[k], (np.ndarray, np.generic)) else sample[k] for k in sample.keys()}
                name = sample['info']['filename'].split('/')[-1]
                if opt.fix_reference:
                    if ref is None:
                        ref = sample['ref-image']
                    else:
                        sample['ref-image'] = ref
                else:
                    ref = sample['ref-image']
                outpath = os.path.join(opt.outdir, name)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model(sample)
                # uc = {'c_concat': [torch.zeros_like(v) for v in c['c_concat']], 'c_crossattn': [torch.zeros_like(v) for v in c['c_crossattn']]} #
                uc = {'c_concat': c['c_concat'], 'c_crossattn': [torch.zeros_like(v) for v in c['c_crossattn']]} #

                shape = (c['c_concat'][0].shape[1],)+c['c_concat'][0].shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c['c_concat'][0].shape[0],
                                                 shape=shape,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                prot_image = torch.clamp((sample['image']+1.0)/2.0,
                                    min=0.0, max=1.0)
                ref_image = torch.clamp((ref+1.0)/2.0,
                                    min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                if not opt.fix_reference:
                    prot_image = prot_image.cpu().numpy()[0]*255
                    Image.fromarray(prot_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_protein.png')
                
                ref_image = ref_image.cpu().numpy()[0]*255
                Image.fromarray(ref_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_reference.png')
                
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(predicted_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_prediction.png')

                predicted_images.append(predicted_image)
                locations.append(sample['info']['locations'])

    # plot the first 20 images in a grid
    plt.figure(figsize=(20,12))
    # set color map to gray
    plt.set_cmap('gray')

    for i in range(15):
        plt.subplot(3,5,i+1)
        image = predicted_images[i].sum(axis=2)
        # clip the image to 0-1
        image = np.clip(image, 0, 255) / 255.0
        plt.imshow(image)
        plt.axis('off')
        # plot text in each image with locations
        plt.text(0, 20, locations[i], color='white', fontsize=10)
        
    plt.suptitle(f'Stable Diffusion Predicted Images (condition: location, guidance scale = {opt.scale})')
    plt.savefig(os.path.join(opt.outdir, f'predicted-image-grid-s{opt.scale}.png'))

    if opt.fix_reference:
        # Save images in a pickle file
        import pickle
        pickle.dump({"prediction": predicted_images, "reference": ref_image, "locations": locations}, open(os.path.join(opt.outdir, 'predictions.pkl'), 'wb'))
        
        # for each predicted image, we assign a random color map from matplotlib
        # then we stack all the color images into one
        # and save it
        import matplotlib.pyplot as plt
        import random
        import colorsys
        # get all the color maps
        cms = list(plt.cm.datad.keys()) # 75 color maps
        
        result_image = np.zeros((predicted_images[0].shape[0], predicted_images[0].shape[1], 3))
        # Get the color map by name:
        for i in range(len(predicted_images)):
            h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
            r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
            image = predicted_images[i].sum(axis=2)
            # clip the image to 0-1
            image = np.clip(image, 0, 255) / 255.0
            colored_image = np.stack([image*r, image*g, image*b], axis=2)
            result_image += colored_image
        result_image = result_image/result_image.max() * 255.0
        Image.fromarray(result_image.astype(np.uint8)).save(os.path.join(opt.outdir, 'super-multiplexed.png'))
