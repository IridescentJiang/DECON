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
    ## 渲染normal/depth/mask/disparity
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


def render_semantic(verts, faces, semantic_dir, angle, elev, size, device=device):
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
    
    R, T = look_at_view_transform(2.0, 0, 0)
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


def render_pymafx(start, end, dataset, interval):
    root_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/{dataset}/ortho_{interval}_pymafx"
    size = 512
    img_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/{dataset}/ortho_{interval}"
    
    preview_dir = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/{dataset}/ortho_{interval}_pymafx/preview" # 用来融合预览一下正面图
    os.makedirs(preview_dir, exist_ok=True)
    
    subject_list = sorted(os.listdir(root_dir))

    if "preview" in subject_list:
        subject_list.remove("preview")

    if end is not None:
        subject_list = subject_list[start:end]
    else:
        subject_list = subject_list[start:]

    for subject in tqdm(subject_list):
        print("render", subject, "start...")

        subject_dir = f"{img_dir}/{subject}"

        pymafx_dir = f"{subject_dir}/pymafx"
        normal_dir = f"{pymafx_dir}/normal"
        depth_dir = f"{pymafx_dir}/depth"
        disparity_dir = f"{pymafx_dir}/disparity"
        mask_dir = f"{pymafx_dir}/mask"

        scan_rgb_dir = f"{subject_dir}/scan/rgb"

        semantic_dir = f"{pymafx_dir}/semantic" # 语义图

        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(disparity_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        os.makedirs(semantic_dir, exist_ok=True)

        # dir for reading
        pymafx_obj_path = os.path.join(
            root_dir, f"{subject}/smplx_mesh.obj"
        )

        ### load smplx mesh
        verts, faces, aux = load_obj(pymafx_obj_path, device=device)

        # 旋转一下（pymafx估计的坐标设置不一样） 或者*(1, -1, -1)也可以
        rot = RotateAxisAngle(180, "X", device=device)
        verts = rot.transform_points(verts)
        
        import random
        preview_angle = random.choice([0, 30, 60, 180, 270])

        for angle in tqdm(range(0, 360, interval)):
            normal, depth = render_smplx(verts, faces, normal_dir, depth_dir, disparity_dir, mask_dir, angle, 0, size, device)
            semantic = render_semantic(verts, faces, semantic_dir, angle, 0, size, device)

            if angle  == 0:
                scan_rgb = imageio.imread(os.path.join(scan_rgb_dir, f"{angle:03d}.png"))
                scan_rgb = scan_rgb/255.0
                r1_n2 = (scan_rgb + normal) / 2.0
                n2_s2 = (normal + semantic) / 2.0
                merge = np.concatenate([r1_n2, n2_s2], axis=1)
                save_path = os.path.join(preview_dir, f"{subject}.png")
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
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="THuman2") # THuman2 CustomHumans

    args = parser.parse_args()
    # 初始化随机数生成器的种子
    random.seed(args.begin)
    render_pymafx(args.begin, args.end, args.dataset, args.interval) # 正面为000的渲染

        # render_thuman2_front(526, 2000)

    # CUDA_VISIBLE_DEVICES=0 python render_scripts/render_pymafx.py  --begin 0 --end 600 --interval 6
    # CUDA_VISIBLE_DEVICES=1 python render_scripts/render_pymafx.py  --begin 600 --end 1200 --interval 6
    # CUDA_VISIBLE_DEVICES=2 python render_scripts/render_pymafx.py  --begin 1200 --end 1800 --interval 6
    # CUDA_VISIBLE_DEVICES=3 python render_scripts/render_pymafx.py  --begin 1800 --interval 6