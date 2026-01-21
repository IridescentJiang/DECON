'''
/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/THuman2/ortho
- scan
    - 0000
        - scan
            - rgb
            - normal
            - depth
            - mask
        - smplx
            - normal
            - depth
            - mask
        - valid 验证渲染结果是否对齐的

        elev.txt
        light.txt
'''
import os
import sys
import glob
import torch
import pickle

# import matplotlib.pyplot as plt
import imageio
import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
import torch.nn.functional as F

# from skimage.io import imread, imsave

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from mmhuman3d.core.cameras import WeakPerspectiveCameras

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.transforms import RotateAxisAngle

from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    blending,
    PerspectiveCameras,
)
from PIL import ImageColor, Image
import argparse

# add path for demo utils functions
from tqdm import tqdm

import random
# # 初始化随机数生成器的种子
# random.seed(224)

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


aa_factor = 6

class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(
            texels, fragments, blend_params, znear=-256, zfar=256
        )

        return images
    
import numpy as np


def depth_normalize(fragments):
    depth_map = fragments.zbuf[0].cpu().numpy()
    # 注意 这其中背景被填充为-1 前景用fragments.pix_to_face > 0标识
    foreground_mask = fragments.pix_to_face[0] > 0
    foreground_mask = foreground_mask.cpu().numpy()
    background_mask = ~foreground_mask

    min_val = np.min(depth_map[foreground_mask])
    max_val = np.max(depth_map[foreground_mask])

    normalized_foreground = (depth_map[foreground_mask] - min_val) / (max_val - min_val)

    normalized_depth_map = np.zeros_like(depth_map)
    normalized_depth_map[foreground_mask] = normalized_foreground

    normalized_depth_map[background_mask] = 1

    # normalized_depth_map = 1 - normalized_depth_map

    return normalized_depth_map



def render_smplx(verts, faces, normal_dir, depth_dir, disparity_dir, mask_dir, angle, elev, size, device=device):
    # mesh转azim和elev
    azim_rot = RotateAxisAngle(angle, "Y", device=device)
    verts = azim_rot.transform_points(verts)
    mesh_smplx = Meshes(verts=[verts], faces=[faces.verts_idx])
    mesh_smplx.textures = TexturesVertex(
        verts_features=(mesh_smplx.verts_normals_padded() + 1.0) * 0.5
    )
    
    bg = "black"
    blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

    raster_settings_mesh = RasterizationSettings(
        image_size=size,
        blur_radius=np.log(1.0 / 1e-4) * 1e-7,
        # bin_size=-1,
        bin_size=0,
        faces_per_pixel=1,
    )
    
    R, T = look_at_view_transform(2.0, 0, 0)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    meshRas = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_mesh)

    renderer = MeshRenderer(
        rasterizer=meshRas,
        shader=cleanShader(blend_params=blendparam),
    )

    # normal
    images = renderer(mesh_smplx)
    rendered_image = images[0, ..., :3].detach().cpu().numpy()
    save_path = os.path.join(normal_dir, f"{angle:03d}.png")
    imageio.imsave(save_path, (rendered_image * 255).astype(np.uint8))

    mask = images[0, ..., 3].detach().cpu().numpy()
    mask[mask > 0] = 1
    mask_save_path = os.path.join(mask_dir, f"{angle:03d}.png")
    imageio.imsave(mask_save_path, (255 * mask).astype(np.uint8))

    # depth
    fragments = meshRas(mesh_smplx)
    normalized_depth_map = depth_normalize(fragments)
    
    save_path = os.path.join(depth_dir, f"{angle:03d}.png")
    imageio.imsave(save_path, (normalized_depth_map * 65535).astype(np.uint16))

    disparity_path = os.path.join(disparity_dir, f"{angle:03d}.png")
    imageio.imsave(disparity_path, ((1-normalized_depth_map) * 65535).astype(np.uint16))
    
    return rendered_image, normalized_depth_map
    

def render_pymafx(start, end, dataset, interval):
    root_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/in-the-wild/{dataset}_pymafx"
    size = 512
    img_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/in-the-wild/{dataset}_{interval}"
    
    preview_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/in-the-wild/{dataset}_pymafx/_preview" # 用来融合预览一下正面图
    os.makedirs(preview_dir, exist_ok=True)
    
    for subject in tqdm(range(start, end)):
        print("render", subject, "start...")

        subject_dir = f"{img_dir}/{subject:04d}"

        pymafx_dir = f"{subject_dir}/pymafx"
        normal_dir = f"{pymafx_dir}/normal"
        depth_dir = f"{pymafx_dir}/depth"
        disparity_dir = f"{pymafx_dir}/disparity"
        mask_dir = f"{pymafx_dir}/mask"

        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(disparity_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # dir for reading
        pymafx_obj_path = os.path.join(
            root_dir, f"{subject:04d}/smplx_mesh.obj"
        )


        ### load smplx mesh
        verts, faces, aux = load_obj(pymafx_obj_path, device=device)

        # 旋转一下（pymafx估计的坐标设置不一样） 或者*(1, -1, -1)也可以
        rot = RotateAxisAngle(180, "X", device=device)
        verts = rot.transform_points(verts)
        
        for angle in tqdm(range(0, 360, interval)):
            normal, depth = render_smplx(verts, faces, normal_dir, depth_dir, disparity_dir, mask_dir, angle, 0, size, device)

            if angle  == 0:
                rgb = imageio.imread(pymafx_obj_path.replace("smplx_mesh.obj", "000.png"))
                rgb = rgb/255.0
                merge = (rgb + normal) / 2.0
                save_path = os.path.join(preview_dir, f"{subject:04d}.png")
                imageio.imsave(save_path, (merge * 255).astype(np.uint8))

        # for angle in tqdm(range(0, 360, interval)):
        #     scan_rgb = render_scan_rgb(mesh, scan_rgb_dir, scan_mask_dir, angle, elev, size, device, light_loc = light_loc)
        #     scan_normal = render_scan_normal(mesh, scan_normal_dir, angle, elev, size, device)
        #     scan_depth = render_scan_depth(mesh, scan_depth_dir, angle, elev, size, device)

        #     normal, depth = render_smplx(verts, faces, smplx_normal_dir, smplx_depth_dir, smplx_mask_dir, angle, elev, size, device)


        # for angle in tqdm(range(0, 360, 6)):
        #     rgb = render_rgb(mesh, rgb_save_dir_new, mask_save_dir_new, angle, 0, size, device, light_loc = light_loc)
        #     normal = render_normal(verts, faces, normal_save_dir_new, angle, 0, size, device, scale=scan_scale, transl=offset)
        #     depth = render_depth(verts, faces, depth_save_dir_new, angle, 0, size, device, scale=scan_scale, transl=offset)
        #     if angle % 30 == 0:
        #         merge = (rgb + normal) / 2.0
        #         save_path = os.path.join(merge_save_dir_new, f"{angle:03d}.png")
        #         imageio.imsave(save_path, (merge * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=16)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="econ")

    args = parser.parse_args()
    # 初始化随机数生成器的种子
    random.seed(args.begin)
    render_pymafx(args.begin, args.end, args.dataset, args.interval) # 正面为000的渲染

        # render_thuman2_front(526, 2000)
