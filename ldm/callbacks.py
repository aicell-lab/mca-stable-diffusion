from __future__ import annotations
from collections import defaultdict
import os
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from tabulate import tabulate
from omegaconf import OmegaConf
from PIL import Image
import psutil
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
import wandb

from ldm.evaluation.metrics import calc_metrics
from ldm.util import send_message_to_slack, send_image_to_slack
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_to_slack=None, log_images_kwargs=None, log_validation=False, val_batch_frequency=1, monitor_val_metric=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.log_validation = log_validation
        self.val_batch_frequency = val_batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_to_slack = log_to_slack
        self._last_val_loss = None
        self.monitor_val_metric = monitor_val_metric

    @rank_zero_only
    def _testtube(self, pl_module, images, samples, targets, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _wandb(self, pl_module, images, samples, targets, batch_idx, split):
        for k in images:
            grid = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            image = wandb.Image(grid)

            tag = f"images-{split}/gs{pl_module.global_step}_{k}"
            wandb.log({tag: image}, step=pl_module.global_step)

        mse, ssim = calc_metrics(samples, targets)
        # pl_module.log(f"{split}/mse", mse) # Don't set `on_step` or `on_epoch` since this is already inside `on_train_batch_end()` or `on_validation_batch_end`
        # pl_module.log(f"{split}/ssim", ssim)
        wandb.log({f"{split}/mse": mse, f"{split}/ssim": ssim}, step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
            # Only send validation images to slack
            if split == "val" and self.log_to_slack and k in self.log_to_slack:
                send_image_to_slack(f"A new image is generated by the diffusion model (group: {split}): ", path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        assert split in ['train', 'val']
        if hasattr(pl_module, "gen_images") and callable(pl_module.gen_images) and self.max_images > 0:
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images, targets, samples = pl_module.gen_images(batch, split=split, N=self.max_images, **self.log_images_kwargs)

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, samples, targets, batch_idx, split)

            if is_train:
                pl_module.train()
        
        # add this to use autoencoders logging images
        elif hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0:
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            
            with torch.no_grad():
                img_log = pl_module.log_images(batch)

            for k in img_log:
                N = min(img_log[k].shape[0], self.max_images)
                img_log[k] = img_log[k][:N]
                if isinstance(img_log[k], torch.Tensor):
                    img_log[k] = img_log[k].detach().cpu()
                    if self.clamp:
                        img_log[k] = torch.clamp(img_log[k], -1., 1.)  

            self.log_local(pl_module.logger.save_dir, split, img_log,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            # for the autoencoder samples are not interesting
            # hack the calculate_metrics to return mse and ssim of the reconstructions and inputs
            logger_log_images(pl_module, img_log, img_log['reconstructions'], img_log['inputs'], batch_idx, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print(f"In on_train_batch_end, batch_idx {batch_idx}")
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step) and self.check_frequency(check_idx):
            print(f"Logging training images in batch {batch_idx} at step {pl_module.global_step}")
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print(f"In on_validation_batch_end, batch_idx {batch_idx}")
        if self.log_validation and batch_idx % self.val_batch_frequency == 0 and not self.disabled and pl_module.global_step > 0:
            print(f"Logging validation images in batch {batch_idx}")
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_to_slack and self.monitor_val_metric and self.monitor_val_metric in trainer.logged_metrics:
            val = trainer.logged_metrics[self.monitor_val_metric]
            if self._last_val_loss:
                if self._last_val_loss > val:
                    self._last_val_loss = val
                    send_message_to_slack(f"🎉 Validation metric updated: {self.monitor_val_metric} reached {val}")
            else:
                self._last_val_loss = val


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 30
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Epoch peak GPU memory {max_memory:.2f}GB")
        except AttributeError:
            pass


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res['rss'] += mmap.rss
        res['pss'] += mmap.pss
        res['uss'] += mmap.private_clean + mmap.private_dirty
        res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
        res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
    return res


class CPUMemoryMonitor(Callback):
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)
  
    def on_train_epoch_start(self, trainer, pl_module):
        print("Start of training epoch", self.str())

    def on_train_epoch_end(self, trainer, pl_module):
        print("End of training epoch", self.str())

    def on_validation_epoch_start(self, trainer, pl_module):
        print("Start of validation epoch", self.str())

    def on_validation_epoch_end(self, trainer, pl_module):
        print("End of validation epoch", self.str())


class CustomEarlyStopping(EarlyStopping):
    """Custom early stopping to check more often than after every epoch or after validation"""
    def __init__(self, *args, check_frequency_gs, start_check_from_gs=0, no_val=True,
                  save_callback_in_ckpt=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_frequency_gs = check_frequency_gs 
        self.start_check_from_gs = start_check_from_gs
        self.no_val = no_val
        self.save_callback_in_ckpt = save_callback_in_ckpt

    def on_validation_end(self, trainer, pl_module) -> None:
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.check_frequency_gs == 0) and (pl_module.global_step > self.start_check_from_gs) :
            print('Checking early stop at global step', pl_module.global_step)
            self._run_early_stopping_check(trainer)
            if trainer.should_stop is True:
                if self.no_val:
                    trainer.limit_val_batches = 0 # avoid running validation after stop signal to trainer
                if not self.save_callback_in_ckpt:
                    for i in range(len(trainer.callbacks)): 
                        if trainer.callbacks[i] == self:
                            trainer.callbacks.pop(i) # remove early stopping callback to be able to change the settings for next run if using trainer.resume_from_checkpoint
                            break
                