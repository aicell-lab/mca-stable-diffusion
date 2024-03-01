import torch
import torch.nn.functional as F
import torchmetrics
# from torchmetrics.image.fid import FrechetInceptionDistance


# class ImageEvaluator:
#     def __init__(self):
#         self.fid = FrechetInceptionDistance(normalize=True)

def calc_metrics(samples, targets):
    '''The value range of the inputs should be between -1 and 1.'''
    bs, resolution = targets.size(0), targets.size(2)
    assert targets.size() == (bs, 3, resolution, resolution)
    assert samples.size() == (bs, 3, resolution, resolution)
    assert targets.min() >= -1  
    assert targets.max() <= 1

    targets = (targets + 1) / 2
    samples = (samples + 1) / 2
    samples = torch.clip(samples, min=0, max=1)
    mse = F.mse_loss(samples, targets).item()
    ssim = torchmetrics.functional.image.ssim.ssim(samples, targets).item()
    # self.fid.update(targets, real=True)
    # self.fid.update(samples, real=False)
    # fid = self.fid.compute().item()
    return mse, ssim





if __name__ == "__main__":
    targets = torch.load("/data/xikunz/stable-diffusion/gt_images.pt")
    samples = torch.load("/data/xikunz/stable-diffusion/gen_images.pt")
    mse, ssim = calc_metrics(targets, samples)
    print(mse, ssim)