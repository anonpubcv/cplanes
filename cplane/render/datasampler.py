import math
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from cplane.render.render import OctreeRender_trilinear_fast as renderer
from cplane.render.render import evaluation
from cplane.render.util.Reg import TVLoss, compute_dist_loss
from cplane.render.util.Sampling import GM_Resi, cal_n_samples
from cplane.render.util.util import N_to_reso


class SimpleSamplerInf:
    """
    A sampler that samples a batch of ids randomly.
    """

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


def init_sampler(cfg, train_dataset,device):
        """
        Initialize the sampler for the training dataset.
        """
        global_mean = []
        if cfg.data.datasampler_type == "rays":
            sampler = SimpleSamplerInf(len(train_dataset), cfg.optim.batch_size)
        elif cfg.data.datasampler_type == "images":
            sampler = SimpleSamplerInf(len(train_dataset), 1)
        elif cfg.data.datasampler_type == "hierach":
            global_mean = train_dataset.global_mean_rgb.to(device)
        return sampler,global_mean
    

def sample_data(cfg, train_dataset, iteration,finetune,device):
        """
        Sample a batch of data from the dataset.
        """
    
        sampler,global_mean = init_sampler(cfg, train_dataset,device)
        train_depth = None
        
        # gen_train = allgens
        # sample rays: shuffle all the rays of training dataset and sampled a batch of rays from them.
        if cfg.data.datasampler_type == "rays":
            ray_idx = sampler.nextids()
            data = train_dataset[ray_idx]
            rays_train, rgb_train, frame_time = (
                data["rays"],
                data["rgbs"].to(device),
                data["time"],)
        # sample images: randomly pick one image from the training dataset and sample a batch of rays from all the rays of the image.
        elif cfg.data.datasampler_type == "images":
            img_i = sampler.nextids()
            data = train_dataset[img_i]
            rays_train, rgb_train, frame_time = (
                data["rays"],
                data["rgbs"].to(device).view(-1, 3),
                data["time"],
            )
            select_inds = torch.randperm(rays_train.shape[0])[
                : cfg.optim.batch_size
            ]
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            # gen_train = gen_train[select_inds]
            frame_time = frame_time[select_inds]
           
        elif cfg.data.datasampler_type == "hierach":
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.
            if iteration <= cfg.data.stage_1_iteration:
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(
                    train_dataset.all_rgbs.shape[1] // cfg.data.key_f_num
                )
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i * cfg.data.key_f_num]
                    .view(-1, 3)
                    .to(device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[
                    cam_i, index_i * cfg.data.key_f_num
                ]
                # Calcualte the probability of sampling each ray based on the difference of global median and local values.
                probability = GM_Resi(
                    rgb_train, global_mean[cam_i], cfg.data.stage_1_gamma
                )
                select_inds = torch.multinomial(
                    probability, cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            elif (
                iteration
                <= cfg.data.stage_2_iteration + cfg.data.stage_1_iteration
            ):
                # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(train_dataset.all_rgbs.shape[1])
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(device)
                )

                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
         
                # gen_train = train_dataset.all_gens[cam_i, index_i]
                probability = GM_Resi(
                    rgb_train, global_mean[cam_i], cfg.data.stage_2_gamma
                )
                select_inds = torch.multinomial(
                    probability, cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
               
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            else:
                # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
                # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                N_time = train_dataset.all_rgbs.shape[1]
                # Sample two adjacent time steps within a range of 25 frames.
                index_i = np.random.choice(N_time)
                index_2 = np.random.choice(
                    min(N_time, index_i + 25) - max(index_i - 25, 0)
                ) + max(index_i - 25, 0)
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(device)
                )
                
                rgb_ref = (
                    train_dataset.all_rgbs[cam_i, index_2].view(-1, 3).to(device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
                # gen_train = train_dataset.all_gens[cam_i, index_i]
                # Calcualte the temporal difference between the two frames as sampling probability.
                probability = torch.clamp(
                    1 / 3 * torch.norm(rgb_train - rgb_ref, p=1, dim=-1),
                    min=cfg.data.stage_3_alpha,
                )
                select_inds = torch.multinomial(
                    probability, cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
               
     
        return rays_train, rgb_train, frame_time

def get_lr_decay_factor(cfg, step):
        """
        Calculate the learning rate decay factor = current_lr / initial_lr.
        """
        if cfg.optim.lr_decay_step == -1:
            cfg.optim.lr_decay_step = cfg.optim.n_iters
 
        if  cfg.optim.lr_decay_type == "exp":  # exponential decay
            lr_factor = cfg.optim.lr_decay_target_ratio ** (
                step / cfg.optim.lr_decay_step
            )
        elif  cfg.optim.lr_decay_type == "linear":  # linear decay
            lr_factor =  cfg.optim.lr_decay_target_ratio + (
                1 -  cfg.optim.lr_decay_target_ratio
            ) * (1 - step /  cfg.optim.lr_decay_step)
        elif  cfg.optim.lr_decay_type == "cosine":  # consine decay
            lr_factor =  cfg.optim.lr_decay_target_ratio + (
                1 -  cfg.optim.lr_decay_target_ratio
            ) * 0.5 * (1 + math.cos(math.pi * step /  cfg.optim.lr_decay_step))

        return lr_factor