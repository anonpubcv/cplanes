import torch
from torch.nn import functional as F
import cv2
import imageio
import numpy as np

from cplane.model.Cplane_Base import Cplane_Base

class Cplane(Cplane_Base):
    """
    A general version of Cplane, which supports different fusion methods and feature regressor methods.
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far,gen_latents_lists, **kargs):
        super().__init__(aabb, gridSize, device, time_grid, near_far,gen_latents_lists, **kargs)


        self.gen_latents_lists = gen_latents_lists
        self.device=device
    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        self.density_plane, self.density_line_time = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        self.app_plane, self.app_line_time = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )

        self.density_plane_ori, self.density_line_time_ori = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        self.app_plane_ori, self.app_line_time_ori = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )

        if (
            self.fusion_two != "concat"
        ):  # if fusion_two is not concat, then we need dimensions from each paired planes are the same.
            assert self.app_n_comp[0] == self.app_n_comp[1]
            assert self.app_n_comp[0] == self.app_n_comp[2]

        # We use density_basis_mat and app_basis_mat to project extracted features from Cplane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        if self.fusion_two == "concat":
            if self.fusion_one == "concat":
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp) * 2, self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp) * 2, self.app_dim, bias=False
                ).to(device)
            else:
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp), self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp), self.app_dim, bias=False
                ).to(device)
        else:
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        # Initialize the basis matrices
        with torch.no_grad():
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            self.density_basis_mat.weight.copy_(weights)

    def get_gen_latent_list(self):
        return self.gen_latents_lists
    
    def get_app_regressor(self):
        return self.app_regressor
    
    def get_density_regressor(self):
        return self.density_regressor


    def init_one_hexplane(self, n_component, gridSize, device):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
                    + self.init_shift
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_time_coef
        ).to(device)

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {     "params": self.density_line_time_ori,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane_ori,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_line_time_ori,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.app_plane_ori,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        
        if isinstance(self.delta_regressor_space, torch.nn.Module):
            # print("dleta regrgressorr added to gradient")
            grad_vars += [
                {
                    "params": self.delta_regressor_space.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]

        grad_vars += [
                {
                    "params": self.gen_latents_lists,
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,      
                }
            ]

        return grad_vars
    
    def get_inf_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]
        # grad_vars = []

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars = [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]

        # if isinstance(self.scene_density_regressor, torch.nn.Module):
        #     grad_vars += [
        #         {
        #             "params": self.scene_density_regressor.parameters(),
        #             "lr": lr_scale * cfg.lr_density_nn,
        #             "lr_org": cfg.lr_density_nn,
        #         }
        #     ]

        # if isinstance(self.gen_latents_list, torch.nn.Module):
        grad_vars += [
                {
                    "params": self.gen_latents_lists,
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,      
                }
            ]

        return grad_vars
    def compute_densityfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor , gen_latents: torch.Tensor 
    ) -> torch.Tensor:
        """
        Compuate the density features of sampled points from density Cplane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            density: (N) density of sampled points.
        """
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

          # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        gen_latents1 = torch.squeeze(gen_latents,dim=-1)
        line_gen_coord1 = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_gen_coord1 = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_gen_coord1), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )
   

        gen_latents2 = torch.squeeze(gen_latents,dim=-1)
        line_gen_coord   = (
            torch.stack(
                (
                    gen_latents2[...],
                    gen_latents2[...],# self.vecMode[1]],
                    gen_latents2[...]#, self.vecMode[2]],
                )
            )
        )
        line_gen_coord = (
            torch.stack(
                (gen_latents.expand(3, -1, -1).squeeze(-1), line_gen_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat, gen_feat,gen_space_feat = [], [], [],[]
        # plane_feat, line_time_feat = [], []
        # Extract features from six feature planes. Hi
        for idx_plane in range(len(self.density_plane)):
            gen_feat.append(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    line_gen_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            gen_space_feat.append(
                F.grid_sample(
                    self.density_line_time[idx_plane],
                    line_gen_coord1[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        line_gen_feat,gen_space_feat =  torch.stack(gen_feat, dim=0),torch.stack(gen_space_feat, dim=0)
       
        # Fusion One
        if self.fusion_one == "multiply":
            inter = line_gen_feat*gen_space_feat
            # inter = plane_feat
        else:
            raise NotImplementedError("no such fusion type")
        
       
        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.density_basis_mat(inter.T)  # Feature Projection

        return inter
    
    def compute_densityfeature_ori(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor , gen_latents: torch.Tensor 
    ) -> torch.Tensor:
        """
        Compuate the density features of sampled points from density Cplane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            density: (N) density of sampled points.
        """
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

          # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        gen_latents1 = torch.squeeze(gen_latents,dim=-1)
        line_gen_coord1 = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_gen_coord1 = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_gen_coord1), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        gen_latents2 = torch.squeeze(gen_latents,dim=-1)
        line_gen_coord   = (
            torch.stack(
                (
                    gen_latents2[...],
                    gen_latents2[...],# self.vecMode[1]],
                    gen_latents2[...]#, self.vecMode[2]],
                )
            )
        )
        line_gen_coord = (
            torch.stack(
                (gen_latents.expand(3, -1, -1).squeeze(-1), line_gen_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat, gen_feat,gen_space_feat = [], [], [],[]
        for idx_plane in range(len(self.density_plane_ori)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
          
            plane_feat.append(
                F.grid_sample(
                    self.density_plane_ori[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.density_line_time_ori[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
       
        plane_feat, line_time_feat, = torch.stack(plane_feat, dim=0), torch.stack(line_time_feat, dim=0)
        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat* line_time_feat
            # inter = plane_feat
        else:
            raise NotImplementedError("no such fusion type")
        

        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.density_basis_mat(inter.T)  # Feature Projection

        return inter

    def init_delta_regressors(self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor,gen_latents: torch.Tensor ):
        xyz_sampled_new=self.delta_regressor_space(xyz_sampled,frame_time,gen_latents  )
        xyz_sampled_new_new= xyz_sampled_new.detach().clone().requires_grad_(True)
        return xyz_sampled_new_new,frame_time
    

    def compute_appfeature_ori(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor ,gen_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Compuate the app features of sampled points from appearance Cplane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        # print("xyz_sampled frame time gen_latents", xyz_sampled.shape,frame_time.shape, gen_latents.shape)
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )


        line_gen_coord1 = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_gen_coord1 = (
            torch.stack(
                (gen_latents.expand(3, -1, -1).squeeze(-1), line_gen_coord1), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        
        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.app_plane_ori)):
           
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.app_plane_ori[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.app_line_time_ori[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
           

        plane_feat, line_time_feat = torch.stack(plane_feat), torch.stack(
            line_time_feat
        )
        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
            # inter = plane_feat 
        else:
            raise NotImplementedError("no such fusion type")
        
        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.app_basis_mat(inter.T)  # Feature Projection

        return inter

    def compute_appfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor ,gen_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Compuate the app features of sampled points from appearance Cplane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        gen_latents1 = torch.squeeze(gen_latents,dim=-1)
        line_gen_coord1 = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_gen_coord1 = (
            torch.stack(
                (gen_latents.expand(3, -1, -1).squeeze(-1), line_gen_coord1), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        gen_latents2 = torch.squeeze(gen_latents,dim=-1)
        line_gen_coord   = (
            torch.stack(
                (
                    gen_latents2[...],
                    # , self.vecMode2[0]],
                    gen_latents2[...],# self.vecMode[1]],
                    gen_latents2[...]#, self.vecMode[2]],
                )
            )
        )
        line_gen_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_gen_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )
       
        gen_feat,gen_space_feat =  [],[]
        for idx_plane in range(len(self.app_plane)):
            gen_feat.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    line_gen_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            gen_space_feat.append(
                F.grid_sample(
                    self.app_line_time[idx_plane],
                    line_gen_coord1[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )



        line_gen_feat,gen_space_feat = torch.stack(
            gen_feat
        ),torch.stack(
            gen_space_feat
        )

        # # Fusion One
        if self.fusion_one == "multiply":
            inter =  line_gen_feat*gen_space_feat
            # inter = plane_feat 
        else:
            raise NotImplementedError("no such fusion type")
        

        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.app_basis_mat(inter.T)  # Feature Projection

        return inter
            
    
    
        
    def TV_loss_density_ori(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane_ori)):
            total = (
                total + reg(self.density_plane_ori[idx]) + reg2(self.density_line_time_ori[idx])
            )
        return total

    def TV_loss_app_ori(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane_ori)):
            total = total + reg(self.app_plane_ori[idx]) + reg2(self.app_line_time_ori[idx])
        return total

    def L1_loss_density_ori(self):
        total = 0
        for idx in range(len(self.density_plane_ori)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane_ori[idx]))
                + torch.mean(torch.abs(self.density_line_time_ori[idx]))
            )
        return total

    def L1_loss_app_ori(self):
        total = 0
        for idx in range(len(self.density_plane_ori)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane_ori[idx]))
                + torch.mean(torch.abs(self.app_line_time_ori[idx]))
            )
        return total
    
    
    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx]) + reg2(self.density_line_time[idx])
            )
        return total

    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) + reg2(self.app_line_time[idx])
        return total

    def L1_loss_density(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        return total

    def L1_loss_app(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, line_time_coef, res_target, time_grid):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
            line_time_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_time_coef[i].data,
                    size=(res_target[vec_id], time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef, line_time_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
                )
        self.app_plane_ori, self.app_line_time_ori = self.up_sampling_planes(
            self.app_plane_ori, self.app_line_time_ori, res_target, time_grid
        )
        self.density_plane_ori, self.density_line_time_ori = self.up_sampling_planes(
            self.density_plane_ori, self.density_line_time_ori, res_target, time_grid
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")
