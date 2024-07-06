import datetime
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from cplane.dataloader import get_test_dataset, get_train_dataset
from cplane.model import init_model,init_model_fine
from cplane.render.render import evaluation, evaluation_path

from cplane.render.trainer import Trainer
from concat_dataset import concat_data4,concat_data_ndc,concat_data_ndc_2
from tqdm.auto import tqdm
import timm 
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def render_test(cfg):
    cfg.data.datadir = "./nerf/nerf_synthetic/nerf_synthetic/mic"
    train_dataset = get_train_dataset(cfg,name="none", is_stack=False)
       
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg

    if not os.path.exists(cfg.systems.ckpt):
        print("the ckpt path does not exists!!")
        return

    Cplane = torch.load(cfg.systems.ckpt, map_location=device)
    torch.save(Cplane, './model_DNERF.pt')
    logfolder = os.path.dirname(cfg.systems.ckpt)
 
    gen_latents_lists_new = Cplane.get_gen_latent_list()
    aabb = test_dataset.scene_bbox.to(device)

    Cplane, reso_cur = init_model_fine(cfg, aabb, train_dataset.near_far, gen_latents_lists_new, device)
    checkpoint = torch.load("./logs/dnerf-20240624-211557/state_dict.th")
    Cplane.load_state_dict(checkpoint['model_state_dict'])
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
   

    gen_latents_lists_new_test= gen_latents_lists_new[2]


    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all_mic", exist_ok=True)
        evaluation(
            test_dataset,
            Cplane,
            gen_latents_lists_new_test,
            cfg,
            f"{logfolder}/imgs_test_all_mic/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


 

    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all_mic", exist_ok=True)
        evaluation_path(
            test_dataset,
            Cplane,
            gen_latents_lists_new_test,
            cfg,
            f"{logfolder}/imgs_path_all_mic/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


def reconstruction(cfg):
    if cfg.data.datasampler_type == "rays":
        train_dataset1 = get_train_dataset(cfg, is_stack=False)
        cfg.data.datadir =  "./Neural_3D_Video/coffee_martini"
        train_dataset2 = get_train_dataset(cfg, is_stack=False)
        cfg.data.datadir = "./D-NeRF_ori/data/mutant"
        train_dataset3 = get_train_dataset(cfg, is_stack=False)
        cfg.data.datadir = "./D-NeRF_ori/data/trex"
        train_dataset4 = get_train_dataset(cfg, is_stack=False)
        cfg.data.datadir = "./D-NeRF_ori/data/jumpingjacks"
        default = get_train_dataset(cfg, is_stack=False)
        train_dataset =concat_data4(default,train_dataset1,train_dataset2,train_dataset3,train_dataset4,cfg.data.downsample,False,cfg.data.N_vis,cfg.data.time_scale,cfg.data.scene_bbox_min,cfg.data.scene_bbox_max,cfg.data.N_random_pose)
    else:
        train_dataset1 = get_train_dataset(cfg,name="salmon", is_stack=True)
        cfg.data.datadir =  ".Datasets_3D/Neural_3D_Video/coffee_martini" 
        train_dataset2 = get_train_dataset(cfg,name="steak", is_stack=True)
        cfg.data.datadir = "./Datasets_3D/Neural_3D_Video/flame_salmon_1" 
        default = get_train_dataset(cfg,name="salmon", is_stack=True)
        
        train_dataset =concat_data_ndc_2(default,train_dataset1,train_dataset2,cfg.data.downsample,True,cfg.data.N_vis,cfg.data.time_scale,cfg.data.scene_bbox_min,cfg.data.scene_bbox_max,cfg.data.N_random_pose)

    cfg.data.datadir =  "./Neural_3D_Video/flame_salmon_1" 

    test_dataset = get_test_dataset(cfg, is_stack=True)
   
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    near_far = test_dataset.near_far

    if cfg.systems.add_timestamp:
        logfolder = f'{cfg.systems.basedir}/{cfg.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # init model.
    aabb = train_dataset1.scene_bbox.to(device)

    if cfg.data.dataset_name == "dnerf":
        class1_raysize = range(len(train_dataset1.all_times))
        class2_raysize = range(len(train_dataset2.all_times))
        class3_raysize = range(len(train_dataset3.all_times))
        class4_raysize = range(len(train_dataset4.all_times))
    
        class_mapping1 = dict([ (name, [i, 0]) for i, name in enumerate(class1_raysize)] )
        class_mapping2 = dict([ (name+len(train_dataset1.all_times), [len(train_dataset1.all_times)+i, 1]) for i, name in enumerate(class2_raysize) ])
        class_mapping3 = dict([ (name+len(train_dataset1.all_times)+len(train_dataset2.all_times), [len(train_dataset1.all_times)+len(train_dataset2.all_times)+i, 2]) for i, name in enumerate(class3_raysize) ])
        class_mapping4 = dict([ (name+len(train_dataset1.all_times)+len(train_dataset2.all_times)+len(train_dataset3.all_times), [len(train_dataset1.all_times)+len(train_dataset2.all_times)+len(train_dataset3.all_times)+i, 3]) for i, name in enumerate(class4_raysize) ])
        
    else:
        class1_raysize = range(len(train_dataset1))
        class2_raysize = range(len(train_dataset2))
        rank1 = len(train_dataset1)
        rank2 = len(train_dataset2)
        class_mapping1 = dict([ (name, [i, 0]) for i, name in enumerate(class1_raysize)] )
        class_mapping2 = dict([ (name+rank1, [rank1+i, 1]) for i, name in enumerate(class2_raysize) ])

    def merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res
   
    class_mapping = merge(class_mapping1, class_mapping2)
    sorted_class_mapping = {}
    extras= {}
    raw_class_list = []
    for key in sorted(class_mapping.keys()):
        sorted_class_mapping[key] = class_mapping[key]
        raw_class_list.append(class_mapping[key])
    extras["raw_class_mapping"] = sorted_class_mapping
    print("clas map len is",len( extras["raw_class_mapping"]))
    
    all_classes = sorted(
        list(set([classid for view, classid in raw_class_list]))
    )
    extras["raw_class_mapping"] = sorted_class_mapping

    all_classes = sorted(
        list(set([classid for view, classid in raw_class_list]))
    )
    extras["raw_classes"] =    all_classes
    class_to_classid_all = dict(
        [(classid, i) for i, classid in enumerate(all_classes)]
    )
   
    extras["raw_class_list"] = raw_class_list
    extras["raw_class_mapping"] = sorted_class_mapping
    extras["imageid_to_classid"] = [
            class_to_classid_all[classid] for view, classid in raw_class_list
        ]
    # extras
    gen_latents_lists_all = [
        torch.zeros(1)#args.gen_latent_size)
        for _ in range(len(extras["raw_classes"])) ]


    print("gen_latents_lists_all",len(gen_latents_lists_all))
   
    Cplane, reso_cur = init_model(cfg, aabb, near_far, gen_latents_lists_all, device)
 
    
    for latent in gen_latents_lists_all:
        latent.requires_grad = True
   
    trainer = Trainer(
        Cplane,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        extras,
        gen_latents_lists_all,
        summary_writer,
        logfolder,
        device,
    )

    optimizer,loss = trainer.train()

    train_latents = torch.zeros(0)
    test_latents= torch.zeros(0)
    gen_all_latents= torch.zeros(0)
    gen_latents_lists_all = Cplane.get_gen_latent_list()
    torch.save(Cplane, f"{logfolder}/{cfg.expname}.th")
    torch.save({
            'epoch': 5,
            'model_state_dict': Cplane.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'gen_latents':gen_latents_lists_all,
            'loss': loss
          
            }, f"{logfolder}/state_dict.th")
    torch.save(Cplane, f"{logfolder}/{cfg.expname}.th")

    gen_latents_lists_a = torch.stack(gen_latents_lists_all, dim=0)  
    imageid_to_classid = torch.tensor(
    extras["imageid_to_classid"]
    )
    train_latents=gen_latents_lists_a[imageid_to_classid]
    
    torch.save(train_latents, f"{logfolder}/gen_latent_codes_train.th")      
    imageid_to_classid = torch.tensor(
    extras["imageid_to_classid"]
    )
    test_latents=gen_latents_lists_a[imageid_to_classid]

    torch.save(gen_all_latents, f"{logfolder}/gen_latent_codes_all.th")
   
    print("gen all train",train_latents.shape)
    print("gen all test",test_latents.shape)
    print("gen all ",gen_latents_lists_a.shape)
    # Render training viewpoints.
    if cfg.render_train:
        os.makedirs(f"{logfolder}/{cfg.expname}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            Cplane,
            train_latents,
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


    # Render test viewpoints.
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            Cplane,
            gen_latents_lists_a[0],
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    # Render validation viewpoints.
    breakpoint()
    if cfg.render_path:
        os.makedirs(f"{logfolder}/{cfg.expname}/imgs_path_all/", exist_ok=True)
        evaluation_path(
            test_dataset,
            Cplane,
            gen_latents_lists_a[0],
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_path_all/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


if __name__ == "__main__":
    # Load config file from base config, yaml and cli.
    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only and (cfg.render_test or cfg.render_path):
        # Inference only.
        render_test(cfg)
    else:
        # Reconstruction and Inference.
        reconstruction(cfg)
