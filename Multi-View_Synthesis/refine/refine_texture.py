import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from src.utils.util import get_camera
from src.utils.util import (
    save_videos_grid,
    pil_list_to_tensor,
)
import sys
sys.path.append("./thirdparties/econ")
from thirdparties.econ.lib.common.mesh_utils import (
    TexturedMeshRenderer, load_human_nvs_results,
    save_optimed_video, save_optimed_smpl_param, save_optimed_mesh,
)
from thirdparties.econ.lib.dataset.mesh_util import SMPLX, get_visibility
import trimesh
import tqdm
from src.utils.util import save_videos_grid

def save_image(image, path):
    # input:
    #     image: Tensor [1, 3, H, W] [0,1]
    img = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    
def save_mesh(verts, faces, colors, path):
    verts = verts.squeeze(0).detach().cpu().numpy()
    faces = faces.squeeze(0).detach().cpu().numpy()
    colors = (colors.squeeze(0).detach().cpu().numpy()*255).astype(np.uint8)
    
    verts_copy = np.zeros_like(verts)
    verts_copy[:,0] = verts[:,0]
    verts_copy[:,1] = verts[:,1]
    verts_copy[:,2] = verts[:,2]
    mesh = trimesh.Trimesh(
            vertices=verts_copy,
            faces=faces,
            vertex_colors=colors,
        )
    trimesh.repair.fix_inversion(mesh)
    mesh.export(path)
    
def parse_args():
    parser = argparse.ArgumentParser()
    # 默认
    parser.add_argument("--mesh_path", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set/0520/mv/optmized_gen/neus/mesh_textured.obj")
    parser.add_argument("--ref_path", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set/0520/ref/rgb.png")
    parser.add_argument("--nvs_dir", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set/0520/mv/optmized_gen/rgb")
    parser.add_argument("--view_num", type=int, default=20)
    # parser.add_argument("-W", type=int, default=1024)
    # parser.add_argument("-H", type=int, default=1024)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    # 修改
    parser.add_argument("--iters1", type=int, default=1500, help="optimization step") ## 低分辨率跑1500
    parser.add_argument("--iters2", type=int, default=500, help="optimization step") ##d 高分辨率跑500
    parser.add_argument('--save_iter', type=int, default=1000)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = args.device

    ## 准备mesh及可优化参数
    mesh = trimesh.load(args.mesh_path)
    vertices, faces, vertex_colors = mesh.vertices, mesh.faces, mesh.visual.vertex_colors
    
    vertices = torch.from_numpy(vertices).contiguous().float().unsqueeze(0).to(device) # B N 3
    faces = torch.from_numpy(faces).contiguous().unsqueeze(0).to(device) # B F 3
    vertex_colors = torch.from_numpy(vertex_colors/255.0).contiguous().float().unsqueeze(0).to(device) # B N 4
    
    optimed_vertex_colors = vertex_colors[:,:,:3].clone() # B N 3 [0,1]
        
    vertices.requires_grad_(True)
    # faces.requires_grad_(True)
    optimed_vertex_colors.requires_grad_(True)
    ## 准备优化器和策略
    optimizer_mesh = torch.optim.Adam([optimed_vertex_colors, ], lr=8e-5, amsgrad=True)
    
    # scheduler_mesh = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer_mesh,
    #     mode="min",
    #     factor=0.5,
    #     verbose=0,
    #     min_lr=1e-5,
    #     patience=args.patience,
    # )
    
    ## 准备lpips
    lpips_meter = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    ## 准备加权
    lpips_weights = torch.tensor(
        [2.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 1.0, 1.0, 1.0, 1.0, 
         2.0, 1.0, 1.0, 1.0, 1.0, 
         2.0, 1.0, 1.0, 1.0, 1.0, ]
    ).to(device)
    mse_weights = torch.tensor(
        [5.0, 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0, 
         5.0, 0.0, 0.0, 0.0, 0.0, ]
    ).to(device)
    
    ## 准备数据（低分辨率）
    # imgs_np, masks_np, azims_np, elevs_np = load_human_nvs_results(args.nvs_dir, args.ref_path, imSize=[args.W, args.H], view_num=args.view_num)
    imgs_np, azims_np, elevs_np = load_human_nvs_results(args.nvs_dir, args.ref_path, imSize=[args.W, args.H], view_num=args.view_num)
    imgs = torch.from_numpy(imgs_np/255.).float().to(device)
    imgs = imgs.permute(0, 3, 1, 2) # B 3 H W [0,1]
    # masks = torch.from_numpy(masks_np).to(device)
    azims = torch.from_numpy(azims_np).float().to(device)
    elevs = torch.from_numpy(elevs_np).float().to(device)
    # masks = masks.unsqueeze(-1)        

    ## 准备mesh渲染器（低分辨率）
    renderer = TexturedMeshRenderer(size=(args.W , args.H), device=device)
    

    save_mesh(vertices, faces, optimed_vertex_colors, f"tmp/mesh_init.obj" )
    
    ## 低分辨率开始训练 (好像也不用mask 反正背景都是黑的)
    pbar = tqdm.trange(args.iters1)
    
    for i in pbar: # 每个iter都用正面+背面+任意其他shijiao 
        losses = []
        index = np.random.choice([i for i in range(20) if i not in [0, 5, 10, 15]])
        indices = [index, 0, 5, 10, 15] # 随机帧和正背面
        
        for index in indices:
            # 加载gt
            img_gt = imgs[index].unsqueeze(0)
            azim, elev = azims[index].unsqueeze(0), elevs[index].unsqueeze(0)
            # 加载mesh,设置相机,准备渲染
            renderer.load_mesh(vertices, faces, optimed_vertex_colors)
            renderer.set_cameras(-azim, elev)
            img_pred, mask_pred = renderer.render_mesh(bg="white", return_mask=True)
            mask_pred = mask_pred.unsqueeze(1).repeat(1,3,1,1)
            
            # 选择对应的权重
            w_lpips = lpips_weights[index]
            w_mse = mse_weights[index]
            # 计算loss
            mse_loss = F.mse_loss(img_pred[mask_pred], img_gt[mask_pred])
            lpips_loss = lpips_meter(img_gt.clamp(0, 1), img_pred.clamp(0, 1))
            loss = mse_loss * w_mse + lpips_loss * w_lpips
            # loss = mse_loss * w_mse
            
            losses.append(loss)
            
            if index == 0:
                mse_log = mse_loss.item()
                lpips_log = lpips_loss.item()

        # 正面+背面+随机面的loss求和
        loss = sum(losses)
        # * 10.0
        loss.backward()
        optimizer_mesh.step()
        optimizer_mesh.zero_grad()
        pbar.set_description(f"MSE = {mse_log:.6f}  LPIPS = {lpips_log:.6f}")
        # pbar.set_description(f"MSE = {mse_log:.6f}")
        
        ## validate
        if i % 500 == 0:
            with torch.no_grad():
                renderer.load_mesh(vertices, faces, optimed_vertex_colors)
                renderer.set_cameras(-azims, elevs)
                video_pred = renderer.render_mesh(bg="white", return_mask=False)
                video = torch.stack(
                        [video_pred.permute(1,0,2,3).detach().cpu(), 
                        imgs.permute(1,0,2,3).detach().cpu()], dim=0
                        )
                save_videos_grid(video.clamp(0, 1), f"tmp/video_{i}.mp4", n_rows=2, fps=5)
                save_mesh(vertices, faces, optimed_vertex_colors.clamp(0, 1), f"tmp/mesh_{i}.obj" )

    
     ## 准备数据（高分辨率）
    imgs_np, azims_np, elevs_np = load_human_nvs_results(args.nvs_dir, args.ref_path, imSize=[2*args.W, 2*args.H], view_num=args.view_num)
    imgs = torch.from_numpy(imgs_np/255.).float().to(device)
    imgs = imgs.permute(0, 3, 1, 2) # B 3 H W [0,1]
    azims = torch.from_numpy(azims_np).float().to(device)
    elevs = torch.from_numpy(elevs_np).float().to(device)

    ## 准备mesh渲染器（高分辨率）
    renderer = TexturedMeshRenderer(size=(2*args.W , 2*args.H), device=device)
    
    ## 低分辨率开始训练 (好像也不用mask 反正背景都是黑的)
    pbar = tqdm.trange(args.iters1, args.iters1+args.iters2)
    
    for i in pbar: # 每个iter都用正面+背面+任意其他shijiao 
        losses = []
        index = np.random.choice([i for i in range(20) if i not in [0, 5, 10, 15]])
        indices = [index, 0, 5, 10, 15] # 随机帧和正背面
        
        for index in indices:
            # 加载gt
            img_gt = imgs[index].unsqueeze(0)
            azim, elev = azims[index].unsqueeze(0), elevs[index].unsqueeze(0)
            # 加载mesh,设置相机,准备渲染
            renderer.load_mesh(vertices, faces, optimed_vertex_colors)
            renderer.set_cameras(-azim, elev)
            img_pred, mask_pred = renderer.render_mesh(bg="white", return_mask=True)
            mask_pred = mask_pred.unsqueeze(1).repeat(1,3,1,1)
            
            # 选择对应的权重
            w_lpips = lpips_weights[index]
            w_mse = mse_weights[index]
            # 计算loss
            mse_loss = F.mse_loss(img_pred[mask_pred], img_gt[mask_pred])
            lpips_loss = lpips_meter(img_gt.clamp(0, 1), img_pred.clamp(0, 1))
            loss = mse_loss * w_mse + lpips_loss * w_lpips
            # loss = mse_loss * w_mse
            
            losses.append(loss)
            
            if index == 0:
                mse_log = mse_loss.item()
                lpips_log = lpips_loss.item()

        # 正面+背面+随机面的loss求和
        loss = sum(losses)
        # * 10.0
        loss.backward()
        optimizer_mesh.step()
        optimizer_mesh.zero_grad()
        pbar.set_description(f"MSE = {mse_log:.6f}  LPIPS = {lpips_log:.6f}")
        # pbar.set_description(f"MSE = {mse_log:.6f}")
        
        ## validate
        if i % 100 == 0:
            with torch.no_grad():
                renderer.load_mesh(vertices, faces, optimed_vertex_colors)
                renderer.set_cameras(-azims, elevs)
                video_pred = renderer.render_mesh(bg="white", return_mask=False)
                video = torch.stack(
                        [video_pred.permute(1,0,2,3).detach().cpu(), 
                        imgs.permute(1,0,2,3).detach().cpu()], dim=0
                        )
                save_videos_grid(video.clamp(0, 1), f"tmp/video_{i}.mp4", n_rows=2, fps=5)
                save_mesh(vertices, faces, optimed_vertex_colors.clamp(0, 1), f"tmp/mesh_{i}.obj" )



if __name__ == "__main__":
    main()

    