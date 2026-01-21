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

import numpy as np
from mmhuman3d.core.renderer.torch3d_renderer.meshes import ParametricMeshes
from mmhuman3d.core.visualization.visualize_smpl import _prepare_colors
import matplotlib.pyplot as plt

color_mapping = {
    # 下面是tab20b色卡
    10:0, 11:1, 23:2, 27:3, # 躯干
    12:4, 19:5, 3:6, 26:7, 16:7, # 左手
    13:8, 20:9, 15:10, 1:11, 18:11, # 右手
    24:12, 7:13, 9:14, 8:14, # 左腿
    2:16, 17:17, 14:18, 22:18, # 右腿
    # 下面是tab20c色卡
    4:20, 5:20, 6:20, 25:20, # 头脸
    21:21 , # 脖子
} # 用tab20b的0~19(15和19没用) + tab20c的0~1(当做19和20) 

def get_seg_colors():
    # tab20b
    cmap = plt.cm.get_cmap("tab20b", 20)
    colors = [cmap(i)[:3] for i in range(20)]
    # tab20c
    cmap = plt.cm.get_cmap("tab20c", 20)
    colors.append(cmap(0)[:3])
    colors.append(cmap(1)[:3])
    # black
    colors = [(0.,0.,0.)] + colors

    return torch.Tensor(colors).float()

def render_smplx(verts, faces, semantic_dir, angle, elev, size, device):
    # mesh转azim和elev
    azim_rot = RotateAxisAngle(angle, "Y", device=device)
    verts = azim_rot.transform_points(verts)

    verts = verts.unsqueeze(0)
    faces = faces.verts_idx.unsqueeze(0)

    ## 准备纹理
    colors_all = _prepare_colors(palette=['white'],
                                 render_choice='part_silhouette', 
                                 num_person=1, 
                                 num_verts=verts.shape[1], 
                                 model_type='smplx')
    colors_all = colors_all.view(-1, verts.shape[1], 3)
    # 进行一下颜色索引映射
    color_mapping_tensor = torch.tensor([color_mapping[i] if i in color_mapping else -1 for i in range(max(color_mapping)+1)])
    # 上面0会被映射为-1(BUG:不知道为啥着色后会有0) 1~27被映射到0~20(其中15 19没用，所以对应20种颜色)
    # 0看起来是脖子 所以修复一下 把0映射到21
    color_mapping_tensor[0] = 21

    B, N, C = colors_all.shape
    colors_all = color_mapping_tensor[colors_all.view(-1).round().long()].view(B, N, C).float()
    colors_all = colors_all + 1 # 0到0 1~27被映射为1~22 (之后对应黑色000+tab20b的20个+tab20c的2个)
    
    meshes = ParametricMeshes(
        verts=verts,
        faces=faces,
        N_individual_overdide=1,
        model_type="smplx",
        use_nearest=True, # 必须True
        vertex_color=colors_all)
    
    bg = "black"
    blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

    raster_settings_mesh = RasterizationSettings(
        image_size=size,
        blur_radius=np.log(1.0 / 1e-4) * 1e-7,
        # bin_size=-1,
        bin_size=0,
        faces_per_pixel=1,
    )
    
    R, T = look_at_view_transform(2.0, elev, 0)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    meshRas = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_mesh)
    renderer = MeshRenderer(
        rasterizer=meshRas,
        shader=cleanShader(blend_params=blendparam),
    )

    # semantic
    images = renderer(meshes)
    images = images[0,...,0].detach().cpu() # H W 这里得到的是单通道的颜色索引 需要转换成rgb
    color = get_seg_colors()
    H, W = images.shape
    images = color[images.view(-1).round().long()].view(H, W, 3)
    rendered_image = images.numpy()
    save_path = os.path.join(semantic_dir, f"{angle:03d}.png")
    imageio.imsave(save_path, (rendered_image * 255).astype(np.uint8))
    
    return rendered_image
    

def render_subject(subject_device_dirs):
    subject, device_id, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size = subject_device_dirs
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    print("render", subject, "start...")
    subject_dir = f"{img_dir}/{subject}"

    smplx_dir = f"{subject_dir}/smplx"
    merge_dir = f"{subject_dir}/merge_normal"

    smplx_semantic_dir = f"{smplx_dir}/semantic"

    elev_path = f"{subject_dir}/elev.txt"

    os.makedirs(smplx_semantic_dir, exist_ok=True)
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
        semantic = render_smplx(verts, faces, smplx_semantic_dir, angle, elev, size, device)

        if angle % 30 == 0:
            scan_rgb = imageio.imread(f"{subject_dir}/scan/rgb/{angle:03d}.png") / 255.0
            r1_n2 = (scan_rgb + semantic) / 2.0
            merge = np.concatenate([r1_n2], axis=1)
            save_path = os.path.join(merge_dir, f"{angle:03d}.png")
            imageio.imsave(save_path, (merge * 255).astype(np.uint8))


def render_subject_single_cuda(subject, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size):
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    print("render", subject, "start...")
    subject_dir = f"{img_dir}/{subject}"

    smplx_dir = f"{subject_dir}/smplx"
    merge_dir = f"{subject_dir}/merge_normal"

    smplx_semantic_dir = f"{smplx_dir}/semantic"

    elev_path = f"{subject_dir}/elev.txt"

    os.makedirs(smplx_semantic_dir, exist_ok=True)
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
        semantic = render_smplx(verts, faces, smplx_semantic_dir, angle, elev, size, device)

        if angle % 30 == 0:
            scan_rgb = imageio.imread(f"{subject_dir}/scan/rgb/{angle:03d}.png") / 255.0
            r1_n2 = (scan_rgb + semantic) / 2.0
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


    ## 多卡
    # subject_device_dirs = [(subject, i % num_processes, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size) for i, subject in enumerate(subject_list)]
    # with Pool(processes=num_processes) as pool:
    #     for _ in tqdm(pool.imap_unordered(render_subject, subject_device_dirs), total=len(subject_list)):
    #         pass

    ## 单卡
    for subject in tqdm(subject_list):
        render_subject_single_cuda(subject, thuman2_root_dir, thuman2_smplx_dir, img_dir, interval, size)
    

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

## 多卡
# CUDA_VISIBLE_DEVICES=0,1,2,3 python render_scripts/render_customhumans_smplx_segmentation.py --begin 0 --interval 5 --num_processes 4
    
## 单卡
# CUDA_VISIBLE_DEVICES=0 python render_scripts/render_customhumans_smplx_segmentation.py --begin 0 --end 320 --interval 5 --num_processes 1
# CUDA_VISIBLE_DEVICES=1 python render_scripts/render_customhumans_smplx_segmentation.py --begin 320 --interval 5 --num_processes 1
# CUDA_VISIBLE_DEVICES=2 python render_scripts/render_customhumans_smplx_segmentation.py --begin 0 --end 320 --interval 6 --num_processes 1
# CUDA_VISIBLE_DEVICES=3 python render_scripts/render_customhumans_smplx_segmentation.py --begin 320 --interval 6 --num_processes 1