# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import glob
from tqdm import tqdm
from .ray_utils import *

import skimage.morphology

# from camera import pose


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized


def resize_disp(disp, H_new, W_new):
    disp_resized = cv2.resize(disp, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    return disp_resized


def poses_avg(poses):
    # hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    # c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    c2w = viewmatrix(vec2, up, center)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def sobel_by_quantile(img_points: np.ndarray, q: float):
    """Return a boundary mask where 255 indicates boundaries (where gradient is
    bigger than quantile).
    """
    dx0 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[1:-1, :-2], axis=-1)
    dx1 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[1:-1, 2:], axis=-1)
    dy0 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[:-2, 1:-1], axis=-1)
    dy1 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[2:, 1:-1], axis=-1)
    dx01 = (dx0 + dx1) / 2
    dy01 = (dy0 + dy1) / 2
    dxy01 = np.linalg.norm(np.stack([dx01, dy01], axis=-1), axis=-1)

    # (H, W, 1) uint8
    boundary_mask = (dxy01 > np.quantile(dxy01, q)).astype(np.float32)
    boundary_mask = np.pad(boundary_mask, ((1, 1), (1, 1)), constant_values=False)[
        ..., None
    ]
    return 1.0 - boundary_mask


class NvidiaDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1,
        is_stack=False,
        N_vis=-1,
        time_scale=1.0,
        hold_every=8,
        use_disp=0,
        use_foreground_mask="motion_masks",
        with_GT_poses=False,
        ray_type="ndc",
    ):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        print("im in")
        self.root_dir = datadir
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = 1
        self.define_transforms()
        self.with_GT_poses = with_GT_poses
        self.use_disp = use_disp
        self.use_foreground_mask = use_foreground_mask
        self.ndc_ray = True
   
        self.ray_type = ray_type

        self.blender2opencv = np.eye(
            4
        )  # np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        print("im in2")
        self.white_bg = False
        
        if ray_type == "contract":
            self.near_far = [0.0, 256]
            self.near=0.0
            self.far=256
            self.scene_bbox = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        else:
            self.near_far = [0.0, 1.0]
            self.near=0.0
            self.far=1.0
            self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        print("im in3 ")
        
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        if self.with_GT_poses:
            poses_bounds = np.load(
                os.path.join(self.root_dir, "poses_bounds.npy")
            )  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, "images/*")))

        # self.foreground_mask_paths = sorted(
        #     glob.glob(os.path.join(self.root_dir, self.use_foreground_mask, "*.png"))
        # )

        if self.with_GT_poses:
            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
            self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        # hwf = poses[:, :, -1]

        # H = 480
        # W = 854
        tmp_img = Image.open(self.image_paths[0]).convert("RGB")
        tmp_img = np.array(tmp_img)
        self.H = tmp_img.shape[0]
        self.W = tmp_img.shape[1]
        # H = 480
        # W = 640
        self.img_wh = np.array([int(self.W / self.downsample), int(self.H / self.downsample)])
        # self.focal = [(854 / 2 * np.sqrt(3)) / float(self.downsample), (854 / 2 * np.sqrt(3)) / float(self.downsample)]
        self.focal = [
            (max(self.H, self.W) / 2 * np.sqrt(3)) / float(self.downsample),
            (max(self.H, self.W) / 2 * np.sqrt(3)) / float(self.downsample),
        ]
        print("focal is",self.focal)
        # # Step 1: rescale focal length according to training resolution
        # H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        # self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        # self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        if self.with_GT_poses:
            # Step 1: rescale focal length according to training resolution
            # self.H, self.W, self.focal1 = poses[
            #     0, :, -1
            # ]  # original intrinsics, same for all images
            # #  focal = poses[0, :, -1]
            # self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
            # # self.focal = [
            # #     self.focal1 * self.img_wh[0] / W,
            # #     self.focal1 * self.img_wh[1] / H,
            # # ]
            # focal = self.focal1 / self.downsample
            # self.focal = [focal, focal]
            # self.W, self.H = self.img_wh
            # print("focal is",self.focal)
            # Step 2: correct poses
            # Original poses has rotation in form "down right back", change to "right up back"
            # See https://github.com/bmild/nerf/issues/34
            poses = np.concatenate(
                [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1
            )
            # (N_images, 3, 4) exclude H, W, focal
            self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

            # Step 3: correct scale so that the nearest depth is at a little more than 1.0
            # See https://github.com/bmild/nerf/issues/34
            near_original = self.near_fars.min()
            if self.ray_type == "ndc":
                scale_factor = near_original * 0.75  # 0.75 is the default parameter
                self.near_fars /= scale_factor
            else:
                scale_factor = np.abs(self.poses[..., 3]).max() * 2.0
            # the nearest depth is at 1/0.75=1.33
            self.poses[..., 3] /= scale_factor

            # build rendering path
            N_views, N_rots = 120, 2
            tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
            up = normalize(self.poses[:, :3, 1].sum(0))
            rads = np.percentile(np.abs(tt), 90, 0)

            self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

            average_pose = average_poses(self.poses)
            dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
            i_test = np.arange(len(self.poses), self.poses.shape[0])
            # img_list = (
            #     i_test
            #     if self.split != "train"
            #     else list(set(np.arange(len(self.poses))))
            # )
            img_list = (list(set(np.arange(len(self.poses))))
            )
            # print("img_list2", len(img_list))
            print("focal is",self.H,self.W, self.focal)
            directions = get_ray_directions_blender(
                self.H, self.W, [(self.focal[0]),(self.focal[1])],
            )

            self.all_poses = []
       
            for i in img_list:
                c2w = torch.FloatTensor(self.poses[i])  # [3, 4]
                c2w[0] = -c2w[0]
                self.all_poses.append(c2w)

                rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
                # rays_o, rays_d = ndc_rays(self.H, self.W, self.focal, self.near, rays_o, rays_d)
                rays_o, rays_d = ndc_rays(self.H, self.W, self.focal[0], 1.0, rays_o, rays_d)
                # self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

       
        # W, H = 640, 480
        # num_images = len(glob.glob(os.path.join(self.root_dir, "images", "*.png")))
        # i_test = np.arange(0, num_images)
   
        # img_list = i_test if self.split != "train" else list(set(np.arange(num_images)))
        # print("img_list 2", len(img_list))
        # use all N_images to train and val
        self.all_rays_train = []
        self.all_rgbs_train = []
        self.all_ts_train = []
        self.all_rays_test = []
        self.all_rgbs_test = []
        self.all_ts_test= []
        # self.all_disps = []
        # self.all_foreground_masks = []
        # self.all_flows_f = []
        # self.all_flow_masks_f = []
        # self.all_flows_b = []
        # self.all_flow_masks_b = []
        # print("imagelis is",len(img_list))
        # img_eval_interval = 1 if self.N_vis < 0 else len(img_list) // self.N_vis)
        # idxs = list(range(0, all_rays.shape[0], img_eval_interval))
        
        img_eval_interval = (
            1 if self.N_vis < 0 else len(img_list) // self.N_vis
        )
        idxs = list(range(0, len(img_list), img_eval_interval))
       
        # for idx, i in enumerate(img_list):
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ): 
            # i = i+1
            # if i !=0:
            if i >-1:
                image_path = self.image_paths[i]

                if self.use_disp:
                    disp_path = os.path.join(
                        self.root_dir, "disp", str(i).zfill(0) + ".npy"
                    )
                    disp_data = np.load(disp_path)
                    disp = torch.from_numpy(resize_disp(disp_data, H, W)).view(
                        -1
                    )  # TODO: check float32 and shape
                # self.all_disps.append(disp)
                c2w = torch.FloatTensor(self.poses[i])  # [3, 4]
                c2w[0] = -c2w[0]
                # self.all_poses2.append(c2w)

                rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
                rays_o, rays_d = ndc_rays(self.H, self.W, self.focal[0], 1.0, rays_o, rays_d)
                # rays_o, rays_d = ndc_rays(self.H, self.W, self.focal, self.near, rays_o, rays_d)
                # if(i%2==0):
                if(i==0 or i==2):
                    self.all_rays_test += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                else:
                    self.all_rays_train += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


                # foreground_mask_path = self.foreground_mask_paths[i]

                img = Image.open(image_path).convert("RGB")
                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                # print("size img_wh",self.img_wh)

                # foreground_mask = Image.open(foreground_mask_path).convert("RGB")
                # if self.downsample != 1.0:
                #     foreground_mask = foreground_mask.resize(self.img_wh, Image.BILINEAR)
                # foreground_mask = self.transform(foreground_mask)  # (3, h, w)

                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                print("i is",i)
                # print("img", img.shape)
                # if(i%2==0):
                if(i==0 or i==2):
                    self.all_rgbs_test += [img]
                else:
                    self.all_rgbs_train += [img]
                cur_time = torch.tensor(float(i) / (len(img_list) - 1)).expand(rays_o.shape[0], 1)
                # if(i%2==0):
                if(i==0 or i==2):
                    self.all_ts_test += [cur_time]
                    print("curr_time is test",cur_time,i)
                else:
                    self.all_ts_train += [cur_time]
                    print("curr_time is train",cur_time,i)
                

                # foreground_mask = foreground_mask.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                # self.all_foreground_masks += [foreground_mask]

    
               
                # print("idx is",idx)
        if(self.split =="train"):
            self.all_rays = self.all_rays_train
            self.all_rgbs = self.all_rgbs_train
            self.all_ts = self.all_ts_train
        else:
            self.all_rays=self.all_rays_test
            self.all_rgbs = self.all_rgbs_test
            self.all_ts = self.all_ts_test

        if not self.is_stack:
            print("im in 4")
            # print("rays,rgbs and ts are",len(self.all_rays),len(self.all_rgbs),len(self.all_ts))
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_ts = torch.cat(self.all_ts, 0)  # (len(self.meta['frames])*h*w)
            
            # print("rays,rgbs and ts are",self.all_rays.shape,self.all_rgbs.shape,self.all_ts.shape)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            
            # self.all_rays = torch.stack(self.all_rays, 0).reshape(
            #     -1, *self.img_wh[::-1], 3
            # )
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]), h, w, 3)
            
            self.all_ts = torch.stack(self.all_ts, 0)
           
          
      
        self.all_ts = self.time_scale * (self.all_ts * 2.0 - 1.0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        
        sample = {
            "rays": self.all_rays[idx],
            "rgbs": self.all_rgbs[idx],
            "time": self.all_ts[idx],
            # "foreground_masks": self.all_foreground_masks[idx],
        }
        

        return sample
