'''
/apdcephfs_cq10/share_1330077/hexu/NVS/test_set/0000
- ref
    - rgb.png
    - mask.png
    - normal.png(marigold)
- cond 
    - gt
        - mesh_normalized.obj
        - semantic/000.png ...
        - normal/000.png ...
        - mask/000.png ...
    - pymafx
    - optimed
- mv
    - gt
        - mesh/000.png ...
        - normal/000.png ...
        - mask/000.png ...
    - gt_gen
    - 
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

def generate_random_light_source():
    def truncated_normal(mean, std, low, high):
        # Generate a truncated normal distribution value
        value = np.random.normal(mean, std)
        while value < low or value > high:
            value = np.random.normal(mean, std)
        return value
    
    # Parameters for elevation and azimuth
    elevation_mean = 0
    elevation_std = 15  # Adjust this value to change the concentration around the mean
    elevation_low = -30
    elevation_high = 70

    azimuth_mean = 0
    azimuth_std = 60  # Adjust this value to change the concentration around the mean
    azimuth_low = -180
    azimuth_high = 180

    # Generate elevation and azimuth angles
    elevation_angle = truncated_normal(elevation_mean, elevation_std, elevation_low, elevation_high)
    azimuth_angle = truncated_normal(azimuth_mean, azimuth_std, azimuth_low, azimuth_high)
    distance = np.random.uniform(1, 3)
    # Convert spherical coordinates to Cartesian coordinates
    elevation_rad = np.radians(elevation_angle)
    azimuth_rad = np.radians(azimuth_angle)

    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = distance * np.sin(elevation_rad)
    z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    
    return (x, y, z), (distance, elevation_angle, azimuth_angle)


def render_scan_rgb(mesh, save_dir, angle_idx, elev, size, device, light_loc=(0.0, 0.0, 1e5)):
    raster_settings = RasterizationSettings(
        image_size=size * aa_factor,
        blur_radius=np.log(1.0 / 1e-4) * 1e-7 * 0.5,
        bin_size=-1,
        faces_per_pixel=1,
    )

    lights = PointLights(
        device=device,
        ambient_color=((1., 1., 1.),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
        location=(light_loc,),
    )

    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

    R, T = look_at_view_transform(2.0, elev, -angle_idx*22.5)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20)
    # cameras = WeakPerspectiveCameras(device=device, R=R, T=T)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )
    images = renderer(mesh, lights=lights)
    images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
    images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC

    images = images.detach().cpu().numpy()[0]
    image = images[..., :3]

    mask = images[..., 3]
    mask[mask > 0] = 1

    save_path = os.path.join(save_dir, f"{angle_idx:02d}.png")
    imageio.imsave(save_path, (255 * image).astype(np.uint8))
    
    return image



import trimesh
def render_subject(subject, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size, thuman2_flag):
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    print("render", subject, "start...")
    subject_dir = f"{img_dir}/{subject}"
    
    scan_dir = f"{subject_dir}/gt"

    scan_rgb_dir = scan_dir

    os.makedirs(scan_rgb_dir, exist_ok=True)

    # os.makedirs(merge_dir, exist_ok=True)

    # dir for reading
    thuman2_obj_path = glob.glob(os.path.join(thuman2_root_dir, subject, "*.obj"))[0]
    smplx_obj_path = glob.glob(os.path.join(thuman2_smplx_dir, subject, "*.obj"))[0]

    # 和syncdreamer一致 设置为30
    elev = 30
    if thuman2_flag:
        front_deg_path = os.path.join(
            thuman2_smplx_dir, f"{subject}/front_azim.txt"
        )
        angle_deg = float(np.loadtxt(front_deg_path))
    
    flag = (thuman2_flag and int(subject)>=526)

    ### load scan mesh
    verts, faces, aux = load_obj(thuman2_obj_path, device=device)
    if flag: # thuman2 526及之后的翻转
            verts = verts[:, [1, 2, 0]]
    if thuman2_flag: # thuman2旋转到正面
        rot = RotateAxisAngle(-angle_deg, "Y", device=device)
        verts = rot.transform_points(verts)
    
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
    
    # 生成随机的光源位置
    light_loc, light_angle = generate_random_light_source()

    # 都已经转到正面了
    # 对于scan：normal是world space，所以rgb、normal、depth都是相机转azim和elev
    # 对于smplx：normal是screen space，所以normal是mesh转azim和elev（即相机不动，mesh转到camera space），depth是相机转（其实depth都可以）
    # 对于normal和depth，是
    
    with torch.no_grad():
        for idx in tqdm(range(16)):
            scan_rgb = render_scan_rgb(mesh, scan_rgb_dir, idx, elev, size, device, light_loc = light_loc)
            
            
import csv

def render_test(start, end, interval, num_processes, dataset_name):
    thuman2_list = []
    customhumans_list = []
    with open("data_split.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
                if row[0]:  # 检查是否为空
                    thuman2_list.append(f"{int(row[0]):04d}")
                if row[1]:
                    customhumans_list.append(row[1])
    if "thuman2" in args.dataset_name:
        subject_list = thuman2_list
    if "customhumans" in args.dataset_name:
        subject_list = customhumans_list
    if dataset_name == "customhumans":
        root_dir = "/apdcephfs_cq10/share_1330077/hexu/data/CustomHumans/CustomHumans/mesh"
        smplx_dir = "/apdcephfs_cq10/share_1330077/hexu/data/CustomHumans/CustomHumans/smplx"
    elif dataset_name == "thuman2":
        root_dir = "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/model"
        smplx_dir = "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/smplx"

    size = 512
    img_dir = "/apdcephfs_cq10/share_1330077/hexu/NVS/0_nvs_baseline_test/syncdreamer"

    subject_list = sorted(subject_list)
    if end is not None:
        subject_list = subject_list[start:end]
    else:
        subject_list = subject_list[start:]

    # subject_device_dirs = [(subject, i % num_processes, root_dir, smplx_dir, img_dir, interval, size) for i, subject in enumerate(subject_list)]
    # with Pool(processes=num_processes) as pool:
    #     for _ in tqdm(pool.imap_unordered(render_subject_mp, subject_device_dirs), total=len(subject_list)):
    #         pass

    thuman2_flag = dataset_name == "thuman2"
    for subject in tqdm(subject_list):
        render_subject(subject, root_dir, smplx_dir, img_dir, interval, size, thuman2_flag)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--interval", type=int, default=22.5)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default='customhumans', help='customhumans or thuman2') #  

    args = parser.parse_args()
    # 初始化随机数生成器的种子
    random.seed(args.begin)
    render_test(args.begin, args.end, args.interval, args.num_processes, args.dataset_name) # 正面为000的渲染

        # render_thuman2_front(526, 2000)


# CUDA_VISIBLE_DEVICES=0 python render_scripts/render_for_syncdreamer.py --begin 0 --end 30
# CUDA_VISIBLE_DEVICES=1 python render_scripts/render_for_syncdreamer.py --begin 30 --end 60
# CUDA_VISIBLE_DEVICES=2 python render_scripts/render_for_syncdreamer.py --begin 60
# CUDA_VISIBLE_DEVICES=3 python render_scripts/render_for_syncdreamer.py --dataset_name customhumans