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
import trimesh
import tqdm
from src.utils.util import save_videos_grid
import os

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
    # parser.add_argument("--root_dir", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set")
    parser.add_argument("--data", type=str, choices=["wild", "gt_gen", "optimized_gen", "optimized_wo_normal_gen", "wild_optimized_wo_normal_gen"], default="gt_gen")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--view_num", type=int, default=20)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    # 修改
    parser.add_argument("--iters1", type=int, default=1800, help="optimization step") ## 低分辨率跑1800
    parser.add_argument("--iters2", type=int, default=200, help="optimization step") ##d 高分辨率跑200
    # parser.add_argument('--save_iter', type=int, default=1000)
    args = parser.parse_args()

    return args

# A100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data gt_gen --begin 0 --end 42 >rf_gt_0.log
# CUDA_VISIBLE_DEVICES=1 python refine_texture.py --data gt_gen --begin 42 --end 84 >rf_gt_1.log
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data gt_gen --begin 84 >rf_gt_2.log

# V100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data optimized_gen --begin 0 --end 30 >rf_op_0.log
# CUDA_VISIBLE_DEVICES=1 python refine_texture.py --data optimized_gen --begin 30 --end 60 >rf_op_1.log
# CUDA_VISIBLE_DEVICES=2 python refine_texture.py --data optimized_gen --begin 60 --end 90 >rf_op_2.log
# CUDA_VISIBLE_DEVICES=3 python refine_texture.py --data optimized_gen --begin 90 >rf_op_3.log

# V100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data wild --begin 0 --end 45 >rf_wild_0.log 开了
# CUDA_VISIBLE_DEVICES=1 python refine_texture.py --data wild --begin 45 --end 90 >rf_wild_1.log
# CUDA_VISIBLE_DEVICES=2 python refine_texture.py --data wild --begin 90 --end 135 >rf_wild_2.log
# CUDA_VISIBLE_DEVICES=3 python refine_texture.py --data wild --begin 135 --end 180 >rf_wild_3.log 开了
# A100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data wild --begin 180 >rf_wild_4.log 开了

# V100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data optimized_wo_normal_gen --begin 0 --end 25 >log/rf_opt_wo_normal_0.log 开了
# CUDA_VISIBLE_DEVICES=1 python refine_texture.py --data optimized_wo_normal_gen --begin 25 --end 50 >log/rf_opt_wo_normal_1.log
# CUDA_VISIBLE_DEVICES=2 python refine_texture.py --data optimized_wo_normal_gen --begin 50 --end 75 >log/rf_opt_wo_normal_2.log
# CUDA_VISIBLE_DEVICES=3 python refine_texture.py --data optimized_wo_normal_gen --begin 75 --end 100 >log/rf_opt_wo_normal_3.log 开了
# A100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data optimized_wo_normal_gen --begin 100 >log/rf_opt_wo_normal_4.log 还在等！！！ 大概吃完饭可以开

# V100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data wild_optimized_wo_normal_gen --begin 0 --end 45 >log/rf_wild_opt_wo_normal_0.log 
# CUDA_VISIBLE_DEVICES=1 python refine_texture.py --data wild_optimized_wo_normal_gen --begin 45 --end 90 >log/rf_wild_opt_wo_normal_1.log
# CUDA_VISIBLE_DEVICES=2 python refine_texture.py --data wild_optimized_wo_normal_gen --begin 90 --end 135 >log/rf_wild_opt_wo_normal_2.log
# CUDA_VISIBLE_DEVICES=3 python refine_texture.py --data wild_optimized_wo_normal_gen --begin 135 --end 180 >log/rf_wild_opt_wo_normal_3.log 
# A100
# CUDA_VISIBLE_DEVICES=0 python refine_texture.py --data wild_optimized_wo_normal_gen --begin 180 >log/rf_wild_opt_wo_normal_4.log 

def main():
    args = parse_args()
    device = args.device
    # root_dir = args.root_dir
    if args.data == "gt_gen" or args.data == "optimized_gen" or args.data == "optimized_wo_normal_gen":
        root_dir = "/apdcephfs_cq10/share_1330077/hexu/NVS/test_set"
    elif args.data == "wild" or args.data == "wild_optimized_wo_normal_gen":
        root_dir = "/apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/test_set"
    else:
        raise NotImplementedError
        
    subjects = sorted(os.listdir(root_dir))
    
    if args.end is not None:
        subjects = subjects[args.begin: args.end]
    else:
        subjects = subjects[args.begin:]
    
    for subject in tqdm.tqdm(subjects):
        subject_dir = os.path.join(root_dir, subject)
        print("【Begin】", subject_dir)

        if args.data == "gt_gen":
            ## gt gen
            mesh_path = os.path.join(subject_dir, "mv/gt_gen/nvs_dual_branch@step2_union_v1/neus/mesh_textured.obj")
            ref_path = os.path.join(subject_dir, "ref/rgb.png")
            nvs_dir = os.path.join(subject_dir, "mv/gt_gen/nvs_dual_branch@step2_union_v1/rgb")
            save_dir = os.path.join(subject_dir, "mv/gt_gen/nvs_dual_branch@step2_union_v1/mesh")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        
        elif args.data == "optimized_gen":            
            # optim gen
            mesh_path = os.path.join(subject_dir, "mv/optmized_gen/neus/mesh_textured.obj")
            ref_path = os.path.join(subject_dir, "ref/rgb.png")
            nvs_dir = os.path.join(subject_dir, "mv/optmized_gen/rgb")
            save_dir = os.path.join(subject_dir, "mv/optmized_gen/mesh")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

        elif args.data == "optimized_wo_normal_gen":            
            # optim gen
            mesh_path = os.path.join(subject_dir, "mv/optimized_wo_normal_gen/neus/mesh_textured.obj")
            ref_path = os.path.join(subject_dir, "ref/rgb.png")
            nvs_dir = os.path.join(subject_dir, "mv/optimized_wo_normal_gen/rgb")
            save_dir = os.path.join(subject_dir, "mv/optimized_wo_normal_gen/mesh")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

        elif args.data == "wild":
            ## wild (optim gen) 改上面的root_dir
            mesh_path = os.path.join(subject_dir, "mv/optmized_gen/neus/mesh_textured.obj")
            ref_path = os.path.join(subject_dir, "ref/rgb.png")
            nvs_dir = os.path.join(subject_dir, "mv/optmized_gen/rgb")
            save_dir = os.path.join(subject_dir, "mv/optmized_gen/mesh")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
        elif args.data == "wild_optimized_wo_normal_gen":
            ## wild (optim gen) 改上面的root_dir
            mesh_path = os.path.join(subject_dir, "mv/optimized_wo_normal_gen/neus/mesh_textured.obj")
            ref_path = os.path.join(subject_dir, "ref/rgb.png")
            nvs_dir = os.path.join(subject_dir, "mv/optimized_wo_normal_gen/rgb")
            save_dir = os.path.join(subject_dir, "mv/optimized_wo_normal_gen/mesh")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
        else:
            raise NotImplementedError


        ## 准备mesh及可优化参数
        mesh = trimesh.load(mesh_path)
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
            [6.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 
            5.0, 0.0, 0.0, 0.0, 0.0, ]
        ).to(device)
        
        ## 准备数据（低分辨率）
        # imgs_np, masks_np, azims_np, elevs_np = load_human_nvs_results(args.nvs_dir, args.ref_path, imSize=[args.W, args.H], view_num=args.view_num)
        imgs_np, masks_np, azims_np, elevs_np = load_human_nvs_results(nvs_dir, ref_path, imSize=[args.W, args.H], view_num=args.view_num)
        imgs = torch.from_numpy(imgs_np/255.).float().to(device)
        imgs = imgs.permute(0, 3, 1, 2) # B 3 H W [0,1]
        masks = torch.from_numpy(masks_np/255.).float().to(device) # B H W
        masks = masks.unsqueeze(1) # B 1 H W
        azims = torch.from_numpy(azims_np).float().to(device)
        elevs = torch.from_numpy(elevs_np).float().to(device)
        # masks = masks.unsqueeze(-1)        

        ## 准备mesh渲染器（低分辨率）
        renderer = TexturedMeshRenderer(size=(args.W , args.H), device=device)
        
        ## 低分辨率开始训练 (前后左右四个面+任意）
        pbar = tqdm.trange(args.iters1)
        
        for i in pbar: # 每个iter都用正面+背面+任意其他shijiao 
            losses = []
            index = np.random.choice([i for i in range(20) if i not in [0, 5, 10, 15]])
            indices = [index, 0, 5, 10, 15] # 随机帧和四个正交面
            
            for index in indices:
                # 加载gt
                img_gt = imgs[index].unsqueeze(0)
                mask_gt = masks[index].unsqueeze(0)
                azim, elev = azims[index].unsqueeze(0), elevs[index].unsqueeze(0)
                # 加载mesh,设置相机,准备渲染
                renderer.load_mesh(vertices, faces, optimed_vertex_colors)
                renderer.set_cameras(-azim, elev)
                img_pred, mask_pred = renderer.render_mesh(bg="white", return_mask=True)
                mask_pred = mask_pred.unsqueeze(1).repeat(1,3,1,1)
                # 选择对应的权重
                w_lpips = lpips_weights[index]
                w_mse = mse_weights[index]
                # 先乘gt的mask(也就是监督图片的mask) 这样渲染结果超出监督图片的部分就不会被优化成白色
                img_pred = img_pred * mask_gt
                img_gt = img_gt * mask_gt
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
            loss.backward()
            optimizer_mesh.step()
            optimizer_mesh.zero_grad()
            pbar.set_description(f"MSE = {mse_log:.6f}  LPIPS = {lpips_log:.6f}")
            
            ## validate
            if i % 600 == 0:
                with torch.no_grad():
                    renderer.load_mesh(vertices, faces, optimed_vertex_colors)
                    renderer.set_cameras(-azims, elevs)
                    video_pred = renderer.render_mesh(bg="white", return_mask=False)
                    video = torch.stack(
                            [video_pred.permute(1,0,2,3).detach().cpu(), 
                            imgs.permute(1,0,2,3).detach().cpu()], dim=0
                            )
                    save_videos_grid(video.clamp(0, 1), f"{save_dir}/video_{i:04d}.mp4", n_rows=2, fps=5)
                    save_mesh(vertices, faces, optimed_vertex_colors.clamp(0, 1), f"{save_dir}/mesh_{i:04d}.obj" )

        
        ## 准备数据（高分辨率）
        imgs_np, masks_np, azims_np, elevs_np = load_human_nvs_results(nvs_dir, ref_path, imSize=[2*args.W, 2*args.H], view_num=args.view_num)
        imgs = torch.from_numpy(imgs_np/255.).float().to(device)
        imgs = imgs.permute(0, 3, 1, 2) # B 3 H W [0,1]
        masks = torch.from_numpy(masks_np/255.).float().to(device) # B H W
        masks = masks.unsqueeze(1) # B 1 H W
        azims = torch.from_numpy(azims_np).float().to(device)
        elevs = torch.from_numpy(elevs_np).float().to(device)
        ## 准备mesh渲染器（高分辨率）
        renderer = TexturedMeshRenderer(size=(2*args.W , 2*args.H), device=device)
        ## 高分辨率开始训练 (只用正背面+任意帧 重点优化正背面）
        pbar = tqdm.trange(args.iters1, args.iters1+args.iters2)
        for i in pbar: # 每个iter都用正面+背面+任意其他shijiao 
            losses = []
            index = np.random.choice([i for i in range(20) if i not in [0, 10, 1,2,3, 17,18,19]]) # 任意帧不选正背面 也不选接近正面的帧
            indices = [index, 0, 10] # 随机帧和正背面
            
            for index in indices:
                # 加载gt
                img_gt = imgs[index].unsqueeze(0)
                mask_gt = masks[index].unsqueeze(0)
                azim, elev = azims[index].unsqueeze(0), elevs[index].unsqueeze(0)
                # 加载mesh,设置相机,准备渲染
                renderer.load_mesh(vertices, faces, optimed_vertex_colors)
                renderer.set_cameras(-azim, elev)
                img_pred, mask_pred = renderer.render_mesh(bg="white", return_mask=True)
                mask_pred = mask_pred.unsqueeze(1).repeat(1,3,1,1)
                # 选择对应的权重
                w_lpips = lpips_weights[index]
                w_mse = mse_weights[index]
                # 先乘gt的mask(也就是监督图片的mask) 这样渲染结果超出监督图片的部分就不会被优化成白色
                img_pred = img_pred * mask_gt
                img_gt = img_gt * mask_gt
                # 计算loss
                mse_loss = F.mse_loss(img_pred[mask_pred], img_gt[mask_pred]) # 这里是取渲染的mask部分计算
                lpips_loss = lpips_meter(img_gt.clamp(0, 1), img_pred.clamp(0, 1))
                loss = mse_loss * w_mse + lpips_loss * w_lpips
                
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
            if i % 200 == 0 or i == (args.iters1+args.iters2-1):
                with torch.no_grad():
                    renderer.load_mesh(vertices, faces, optimed_vertex_colors)
                    renderer.set_cameras(-azims, elevs)
                    video_pred = renderer.render_mesh(bg="white", return_mask=False)
                    video = torch.stack(
                            [video_pred.permute(1,0,2,3).detach().cpu(), 
                            imgs.permute(1,0,2,3).detach().cpu()], dim=0
                            )
                    if i == (args.iters1+args.iters2-1): # 最后一个epoch
                        save_videos_grid(video.clamp(0, 1), f"{save_dir}/video_final.mp4", n_rows=2, fps=5)
                        save_mesh(vertices, faces, optimed_vertex_colors.clamp(0, 1), f"{save_dir}/mesh_final.obj" )
                    else:
                        save_videos_grid(video.clamp(0, 1), f"{save_dir}/video_{i:04d}.mp4", n_rows=2, fps=5)
                        save_mesh(vertices, faces, optimed_vertex_colors.clamp(0, 1), f"{save_dir}/mesh_{i:04d}.obj" )



if __name__ == "__main__":
    main()

    