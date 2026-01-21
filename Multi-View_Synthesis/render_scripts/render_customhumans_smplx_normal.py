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
from multiprocessing import Pool, cpu_count


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
from PIL import ImageColor
import argparse

# add path for demo utils functions
from tqdm import tqdm

import random
# # 初始化随机数生成器的种子
# random.seed(224)

# Setup device
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")


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

def render_smplx(verts, faces, normal_dir, depth_dir, disparity_dir, mask_dir, angle, elev, size, device):
    # mesh转azim 相机转elev
    azim_rot = RotateAxisAngle(angle, "Y", device=device)
    verts = azim_rot.transform_points(verts)
    # elev_rot = RotateAxisAngle(elev, "X", device=device)
    # verts = elev_rot.transform_points(verts)
    mesh_smplx = Meshes(verts=[verts], faces=[faces.verts_idx])
    mesh_smplx.textures = TexturesVertex(
        verts_features=(mesh_smplx.verts_normals_padded() + 1.0) * 0.5
    )
    
    bg = "black"
    blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

    raster_settings_mesh = RasterizationSettings(
        image_size=size,
        blur_radius=np.log(1.0 / 1e-4) * 1e-7,
        bin_size=-1,
        faces_per_pixel=1,
    )
    
    R, T = look_at_view_transform(2.0, elev, 0)
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

    # # mask
    # mask = images[0, ..., 3].detach().cpu().numpy()
    # mask[mask > 0] = 1
    # mask_save_path = os.path.join(mask_dir, f"{angle:03d}.png")
    # imageio.imsave(mask_save_path, (255 * mask).astype(np.uint8))

    # # depth
    # fragments = meshRas(mesh_smplx)
    # normalized_depth_map = depth_normalize(fragments)
    
    # save_path = os.path.join(depth_dir, f"{angle:03d}.png")
    # imageio.imsave(save_path, (normalized_depth_map * 65535).astype(np.uint16))

    # # disparity 
    # disparity_path = os.path.join(disparity_dir, f"{angle:03d}.png")
    # imageio.imsave(disparity_path, ((1-normalized_depth_map) * 65535).astype(np.uint16))
    
    
    return rendered_image
    

def render_subject(subject_device_dirs):
    subject, device_id, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size = subject_device_dirs
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    print("render", subject, "start...")
    subject_dir = f"{img_dir}/{subject}"

    smplx_dir = f"{subject_dir}/smplx"
    merge_dir = f"{subject_dir}/merge_normal"

    smplx_normal_dir = f"{smplx_dir}/normal"
    smplx_depth_dir = f"{smplx_dir}/depth"
    smplx_disparity_dir = f"{smplx_dir}/disparity"
    smplx_mask_dir = f"{smplx_dir}/mask"

    elev_path = f"{subject_dir}/elev.txt"
    light_path = f"{subject_dir}/light.txt"


    os.makedirs(smplx_normal_dir, exist_ok=True)
    os.makedirs(smplx_depth_dir, exist_ok=True)
    os.makedirs(smplx_disparity_dir, exist_ok=True)
    os.makedirs(smplx_mask_dir, exist_ok=True)

    os.makedirs(merge_dir, exist_ok=True)

    # dir for reading
    thuman2_obj_path = glob.glob(os.path.join(thuman2_root_dir, subject, "*.obj"))[0]
    smplx_obj_path = glob.glob(os.path.join(thuman2_smplx_dir, subject, "*.obj"))[0]

    
    # elevation -5~15° / -5~10?
    elev = float(np.loadtxt(elev_path))
    
    ### load scan mesh
    verts, faces, aux = load_obj(thuman2_obj_path, device=device)
        
    tex_maps = aux.texture_images
    if tex_maps is not None and len(tex_maps) > 0:
        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = faces.textures_idx.to(device)  # (F, 3)
        image = list(tex_maps.values())[0].to(device)[None]
        tex = TexturesUV(
            verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
        )

    mesh = Meshes(
        verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
    )
    # scale
    vertices = mesh.verts_list()[0]
    vertices = vertices.cpu().numpy()
    up_axis = 1
    scan_scale = 1.85 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
    mesh.scale_verts_(scan_scale)

    # scale后居中
    vertices = mesh.verts_list()[0]
    vertices = vertices.cpu().numpy()
    center =  (vertices.max(0) + vertices.min(0)) / 2

    offset = torch.tensor(0 - center).to(device)
    mesh.offset_verts_(offset)
    
    ### load smplx mesh
    verts, faces, aux = load_obj(smplx_obj_path, device=device)

    # scale和居中
    mesh_smplx = Meshes(verts=[verts], faces=[faces.verts_idx])
    mesh_smplx.scale_verts_(scan_scale)
    mesh_smplx.offset_verts_(offset)
    verts = mesh_smplx.verts_list()[0]
    faces = faces


    # 都已经转到正面了
    # 对于scan：normal是world space，所以rgb、normal、depth都是相机转azim和elev
    # 对于smplx：normal是screen space，所以normal是mesh转azim和elev（即相机不动，mesh转到camera space），depth是相机转（其实depth都可以）
    # 对于normal和depth，是
    for angle in tqdm(range(0, 360, interval)):
        normal = render_smplx(verts, faces, smplx_normal_dir, smplx_depth_dir, smplx_disparity_dir, smplx_mask_dir, angle, elev, size, device)

        if angle % 30 == 0:
            scan_rgb = imageio.imread(f"{subject_dir}/scan/rgb/{angle:03d}.png") / 255.0
            r1_n2 = (scan_rgb + normal) / 2.0
            merge = np.concatenate([r1_n2], axis=1)
            save_path = os.path.join(merge_dir, f"{angle:03d}.png")
            imageio.imsave(save_path, (merge * 255).astype(np.uint8))


def render_customhumans(start, end, interval, num_processes):
    thuman2_root_dir = "/apdcephfs_cq10/share_1330077/hexu/data/CustomHumans/CustomHumans/mesh"
    thuman2_smplx_dir = "/apdcephfs_cq10/share_1330077/hexu/data/CustomHumans/CustomHumans/smplx"
    size = 512
    img_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/CustomHumans/ortho_{interval}"

    subject_list = sorted(os.listdir(thuman2_root_dir))
    if end is not None:
        subject_list = subject_list[start:end]
    else:
        subject_list = subject_list[start:]

    subject_device_dirs = [(subject, i % num_processes, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size) for i, subject in enumerate(subject_list)]
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(render_subject, subject_device_dirs), total=len(subject_list)):
            pass

    # for subject in tqdm(subject_list):
    #     render_subject(subject, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size, device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--num_processes", type=int, default=2)

    args = parser.parse_args()
    # 初始化随机数生成器的种子
    random.seed(args.begin)
    render_customhumans(args.begin, args.end, args.interval, args.num_processes) # 正面为000的渲染

        # render_thuman2_front(526, 2000)

# CUDA_VISIBLE_DEVICES=2,3 python render_scripts/render_customhumans_smplx_normal.py --begin 0 --interval 5 --num_processes 2