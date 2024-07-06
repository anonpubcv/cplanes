from .dnerf_dataset import DNerfDataset
# from .neural_3D_dataset_NDC import Neural3D_NDC_Dataset
# from .dynamic_scene_data import DySceneDataset
# from .nvidia import NvidiaDataset


def get_train_dataset(cfg,name, is_stack=False,mode="train", ):
    if cfg.data.dataset_name == "dnerf":
        train_dataset = DNerfDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            mode=mode,
            use_disp=True,
            N_random_pose=cfg.data.N_random_pose,
        )
    # elif cfg.data.dataset_name == "dyscene":
    #     train_dataset = NvidiaDataset(
    #     cfg.data.datadir,
    #     split="train",
    #     downsample=cfg.data.downsample,
    #     is_stack=is_stack,
    #     N_vis=cfg.data.N_vis,
    #     time_scale=cfg.data.time_scale,
    #     use_disp=False,
    #     use_foreground_mask=False,
    #     with_GT_poses=True,
    #     ray_type="ndc",
    #     )
    # elif cfg.data.dataset_name == "dyscene2":
    #     train_dataset = DySceneDataset(
    #         cfg.data.datadir,
    #         "train",
    #         cfg.data.downsample,
    #         is_stack=is_stack,
    #         cal_fine_bbox=cfg.data.cal_fine_bbox,
    #         N_vis=cfg.data.N_vis,
    #         time_scale=cfg.data.time_scale,
    #         scene_bbox_min=cfg.data.scene_bbox_min,
    #         scene_bbox_max=cfg.data.scene_bbox_max,
    #         N_random_pose=cfg.data.N_random_pose,
    #     )
    # elif cfg.data.dataset_name == "neural3D_NDC":
    #     train_dataset = Neural3D_NDC_Dataset(
    #         cfg.data.datadir,
    #         "train",
         
    #         cfg.data.downsample,
    #         is_stack=is_stack,
    #         cal_fine_bbox=cfg.data.cal_fine_bbox,
    #         N_vis=cfg.data.N_vis,
    #         time_scale=cfg.data.time_scale,
    #         scene_bbox_min=cfg.data.scene_bbox_min,
    #         scene_bbox_max=cfg.data.scene_bbox_max,
    #         N_random_pose=cfg.data.N_random_pose,
    #         bd_factor=cfg.data.nv3d_ndc_bd_factor,
    #         eval_step=cfg.data.nv3d_ndc_eval_step,
    #         eval_index=cfg.data.nv3d_ndc_eval_index,
    #         sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
    #         mode=mode,
    #         name=name
    #     )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True):
    if cfg.data.dataset_name == "dnerf":
        test_dataset = DNerfDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            mode="train",
            use_disp=False,
            N_random_pose=cfg.data.N_random_pose,
        )
    # elif cfg.data.dataset_name == "dyscene":
    #     test_dataset = NvidiaDataset(
    #     cfg.data.datadir,
    #     split="test",
    #     downsample=cfg.data.downsample,
    #     is_stack=is_stack,
    #     N_vis=cfg.data.N_vis,
    #     use_disp=True,
    #     use_foreground_mask=False,
    #     with_GT_poses=True,
    #     ray_type="ndc",
    #     )
    # elif cfg.data.dataset_name == "neural3D_NDC":
    #     test_dataset = Neural3D_NDC_Dataset(
    #         cfg.data.datadir,
    #         "test",
    #         cfg.data.downsample,
    #         is_stack=is_stack,
    #         cal_fine_bbox=cfg.data.cal_fine_bbox,
    #         N_vis=cfg.data.N_vis,
    #         time_scale=cfg.data.time_scale,
    #         scene_bbox_min=cfg.data.scene_bbox_min,
    #         scene_bbox_max=cfg.data.scene_bbox_max,
    #         N_random_pose=cfg.data.N_random_pose,
    #         bd_factor=cfg.data.nv3d_ndc_bd_factor,
    #         eval_step=cfg.data.nv3d_ndc_eval_step,
    #         eval_index=cfg.data.nv3d_ndc_eval_index,
    #         sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
    #         mode="test",
    #         name="salmon"
    #     )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset
