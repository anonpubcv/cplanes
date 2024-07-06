
import torch

import numpy as np


def concat_data4(defualt,dataset1, dataset2,dataset3,dataset4,downsample,is_stack,N_vis,time_scale,scene_bbox_min,scene_bbox_max,N_random_pose):
 
    dataset=defualt

    print("len(train_dataset4) is 1 ",len(dataset4))
   
    dataset.root_dir = [ dataset1.root_dir, dataset2.root_dir,dataset3.root_dir,dataset4.root_dir]
    dataset.split = [dataset1.split,dataset2.split,dataset3.split,dataset4.split]
    dataset.downsample = downsample
    dataset.img_wh = (int(800 / downsample), int(800 / downsample)),
    dataset.is_stack = is_stack
    dataset.N_vis = N_vis  # evaluate images for every N_vis images

    dataset.time_scale = 3
    dataset.world_bound_scale = 1.1

    dataset.near = 2.0
    dataset.far = 6.0
    dataset.near_far = [2.0, 6.0]

    dataset.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
    dataset.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    dataset.white_bg = True
    dataset.ndc_ray = False
    dataset.depth_data = False
    dataset2.all_times = torch.add(dataset2.all_times ,0)
    dataset.all_times =torch.cat((dataset1.all_times,dataset2.all_times,dataset3.all_times,dataset4.all_times),0)
    dataset.N_random_pose = N_random_pose
    dataset.center =[torch.mean(dataset1.scene_bbox, axis=0).float().view(1, 1, 3),torch.mean(dataset2.scene_bbox, axis=0).float().view(1, 1, 3),torch.mean(dataset3.scene_bbox, axis=0).float().view(1, 1, 3),torch.mean(dataset4.scene_bbox, axis=0).float().view(1, 1, 3)]
    dataset.all_rays =torch.cat((dataset1.all_rays,dataset2.all_rays,dataset3.all_rays,dataset4.all_rays),0)
    dataset.directions =torch.cat( (dataset1.directions, dataset2.directions,dataset3.directions,dataset4.directions),0)
    dataset.intrinsics =torch.cat((dataset1.intrinsics,dataset2.intrinsics,dataset3.intrinsics,dataset4.intrinsics),0)
    dataset.image_paths =[(dataset1.image_paths,dataset2.image_paths,dataset3.image_paths,dataset4.image_paths)]
    dataset.poses =torch.cat(( dataset1.poses, dataset2.poses,dataset3.poses,dataset4.poses),0)

 

    dataset.all_rgbs =torch.cat((dataset1.all_rgbs, dataset2.all_rgbs, dataset3.all_rgbs,dataset4.all_rgbs),0)
    dataset.all_depth =[(dataset1.all_depth,dataset2.all_depth,dataset3.all_depth,dataset4.all_depth)]
    dataset.all_gens =torch.cat((dataset1.all_gens,dataset2.all_gens,dataset3.all_gens,dataset4.all_gens),0)
    dataset.idxs =[(dataset1.idxs,dataset2.idxs,dataset3.idxs,dataset4.idxs)]

    dataset.proj_mat =torch.cat((dataset1.proj_mat,dataset2.proj_mat,dataset3.proj_mat,dataset4.proj_mat),0)
 
    return dataset