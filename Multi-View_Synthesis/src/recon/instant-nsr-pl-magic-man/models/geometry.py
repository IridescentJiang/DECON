from typing import Callable, Sized
from jaxtyping import Float
# PyTorch Tensor type
from torch import Tensor
# Runtime type checking decorator
from typeguard import typechecked as typechecker

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from models.base import BaseModel
from models.utils import scale_anything, get_activation, cleanup, chunk_batch
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from utils.misc import get_rank, broadcast
from systems.utils import update_module_step
from nerfacc import ContractionType
import os

def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')
        self.radius = self.config.radius
        self.contraction_type = None # assigned in system

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax):
        def batch_func(x):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv
    
        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices()) # 得到512^3的sdf值
        
        # marching默认传入的sdf值的坐标范围在[0,1]
        # coarse阶段:在[-1,1]取sdf,值送给marching cube[0,1],scale回[-1,1]就是原始比例
        # fine阶段:在之前得到的vmin和vmax范围取sdf,值送给marching cube[0,1],再scale回[vmin,vmax]所以还是原始比例
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1) # [0,1]变回[-1,1]
        return mesh

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        # 上面在[-1,1]的立方内按照给定的网格分辨率进行marching cude
        # 考虑到物体在[-1,1]范围占的可能很小 所以根据上述coarse得到一个边界
        # 再用边界（放宽一些，但是仍然比之前的cube小）划分同样分辨率的网格
        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine 


@models.register('volume-density')
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config)

    def forward(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type)
        out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1], self.n_output_dims).float()
        density, feature = out[...,0], out
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        if 'feature_activation' in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type)
        density = self.encoding_with_network(points.reshape(-1, self.n_input_dims)).reshape(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        return -density      

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
                
                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                sdf, feature = out[...,0], out
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  

                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        if self.grad_type == 'finite_difference':
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(f"Update finite_difference_eps to {grid_size}")
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")


    def load_ref_shape(self):
        shape_init_config = self.config.shape_init_config
        if os.path.exists(os.path.join(shape_init_config.cond_dir, "mesh_normalized.obj")):
            mesh_path = os.path.join(shape_init_config.cond_dir, "mesh_normalized.obj")
        elif os.path.exists(os.path.join(shape_init_config.cond_dir, "mesh.obj")):
            mesh_path = os.path.join(shape_init_config.cond_dir, "mesh.obj")
        else:
            raise ValueError(f"Mesh dir {shape_init_config.cond_dir} does not contain mesh[.obj] file.")
        
        import trimesh
        mesh = trimesh.load(mesh_path)
        # align to up-z and front-x
        dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        dir2vec = {
            "+x": np.array([1, 0, 0]),
            "+y": np.array([0, 1, 0]),
            "+z": np.array([0, 0, 1]),
            "-x": np.array([-1, 0, 0]),
            "-y": np.array([0, -1, 0]),
            "-z": np.array([0, 0, -1]),
        }
        if (
            shape_init_config.mesh_up not in dirs
            or shape_init_config.mesh_front not in dirs
        ):
            raise ValueError(f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}.")
        if shape_init_config.mesh_up[1] == shape_init_config.mesh_front[1]:
            raise ValueError("shape_init_mesh_up and shape_init_mesh_front must be orthogonal.")
        z_, x_ = (
            dir2vec[shape_init_config.mesh_up],
            dir2vec[shape_init_config.mesh_front],
        )
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)

        # scaling
        if (shape_init_config.geo_prior_type == 'gt_smpl' 
            or shape_init_config.geo_prior_type == 'pymafx_smpl' 
            or shape_init_config.geo_prior_type == 'optimed_smpl'):
            # if shape_init_config.use_orthograph:
            #     scale = 1.
            # else:
            #     fov = self.cfg.fov
            #     scale = 1. / np.tan(np.deg2rad(fov/2.))
            scale = 1.
            mesh.vertices = mesh.vertices / scale * shape_init_config.shape_init_params
        mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

        from pysdf import SDF

        sdf = SDF(mesh.vertices, mesh.faces)

        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
            # add a negative signed here
            # as in pysdf the inside of the shape has positive signed distance
            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                points_rand
            )[..., None]

        self.get_gt_sdf = func

    def initialize_shape(self, run_init=True) -> None:
        shape_init_config = self.config.shape_init_config
        if shape_init_config.shape_init is None and not shape_init_config.force_shape_init: # false & false
            return

        # # do not initialize shape if weights are provided
        # if self.config.weights is not None and not shape_init_config.force_shape_init: # false/true & false
        #     return

        self.get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(shape_init_config.shape_init, str)
        if shape_init_config.shape_init == "ellipsoid":
            assert (
                isinstance(shape_init_config.shape_init_params, Sized)
                and len(shape_init_config.shape_init_params) == 3
            )
            size = torch.as_tensor(shape_init_config.shape_init_params).to(self.rank)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            self.get_gt_sdf = func
        elif shape_init_config.shape_init == "sphere":
            assert isinstance(shape_init_config.shape_init_params, float)
            radius = shape_init_config.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            self.get_gt_sdf = func
        elif shape_init_config.shape_init.startswith("mesh"):
            assert isinstance(shape_init_config.geo_prior_type, str)
            assert isinstance(shape_init_config.shape_init_params, float)

            self.load_ref_shape()

        else:
            raise ValueError(
                f"Unknown shape initialization type: {shape_init_config.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        if run_init and self.training:
            optim = torch.optim.Adam(self.parameters(), lr=1e-3)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=50, verbose=True)
            
            from tqdm import tqdm
            pbar = tqdm(
                range(1000),
                desc=f"Initializing SDF to a(n) {shape_init_config.shape_init}:",
                disable=get_rank() != 0,
            )
            for iter in pbar:
                points_rand = (
                    torch.rand((20000, 3), dtype=torch.float32, device=self.rank) * 2.0 - 1.0
                )
                
                sdf_pred = self.forward(points_rand, with_grad=False, with_feature=False, with_laplace=False)
                sdf_gt = self.get_gt_sdf(points_rand).view(-1)

                if shape_init_config.predict_offset:
                    loss = F.mse_loss(sdf_pred, sdf_gt)
                else:
                    loss = F.l1_loss(sdf_pred, sdf_gt)
                if iter == 0:
                    print('pre-initialization loss:', loss.item())
                pbar.set_postfix({"loss": loss.item()})
                optim.zero_grad()
                loss.backward()
                optim.step()
                # scheduler.step(loss)
                
            print('post-initialization loss:', loss.item())
            # explicit broadcast to ensure param consistency across ranks
            for param in self.parameters():
                broadcast(param, src=0)