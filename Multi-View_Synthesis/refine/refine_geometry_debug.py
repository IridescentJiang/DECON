import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import torch.utils.checkpoint
from tqdm import tqdm
import os
from src.utils.util import get_camera
from src.utils.util import (
    save_videos_grid,
    pil_list_to_tensor,
)
import sys
sys.path.append("./thirdparties/econ")
from thirdparties.econ.lib.common.smpl_utils import (
    SMPLEstimator, SMPLRenderer,
    save_optimed_video, save_optimed_smpl_param, save_optimed_mesh,
)

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

# from thirdparties.econ.lib.dataset.mesh_util import (remesh)
from thirdparties.econ.lib.dataset.mesh_util import (SMPLX,apply_vertex_mask,part_removal,poisson)
from thirdparties.econ.lib.dataset.TestDataset import TestDataset
from pytorch3d.structures import Meshes

import os
from termcolor import colored
import argparse
import trimesh
import numpy as np
import torch
from thirdparties.econ.lib.dataset.mesh_util import SMPLX
from tensorboardX import SummaryWriter


SMPLX_object=SMPLX()

def refine_subject_geometry(subject_dir, trial_name, device):
    
    smpl_renderer = SMPLRenderer(size=512, device=device)
    smpl_renderer.set_cameras([-120.0], [0.0])

    mesh_path = os.path.join(subject_dir, f"mv/gt_gen/{trial_name}/neus/save/it15000-mc512-colored.obj")
    smpl_mesh_path = os.path.join(subject_dir, "cond/gt/mesh_normalized.obj")

    smpl_mesh = trimesh.load(smpl_mesh_path)
    hand_mesh = smpl_mesh.copy()
    
    # mesh = remesh(mesh_path, 0.5, device)
    # mesh = trimesh.load_mesh("./tmp/remesh.obj") # 平滑过一次的
    mesh = trimesh.load_mesh(mesh_path)
    v1, f1 = mesh.vertices*[1,1,-1], mesh.faces
    v2, f2 = hand_mesh.vertices, hand_mesh.faces
    def np2tr(x):
        return torch.from_numpy(x).unsqueeze(0).float().to(device)
    smpl_renderer.load_mesh(np2tr(v1), np2tr(f1), reverse_normal=True)
    # img1, _ = smpl_renderer.render_normal_screen_space(
    #                     bg="black", return_mask=True)
    img1 = smpl_renderer.render_normal(bg="black")
    
    smpl_renderer.load_mesh(np2tr(v2), np2tr(f2))
    # img2, _ = smpl_renderer.render_normal_screen_space(
    #                     bg="black", return_mask=True)
    img2 = smpl_renderer.render_normal(bg="black")
    
    def ts2pil(img, path):
        img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        from PIL import Image
        img = Image.fromarray(img)
        img.save(path)
        
    ts2pil(img1, "./tmp/img1.png")
    ts2pil(img2, "./tmp/img2.png")
    
    
    hand_mask = torch.zeros(SMPLX_object.smplx_verts.shape[0], )
    hand_mask.index_fill_(
        0, torch.tensor(SMPLX_object.smplx_mano_vid_dict["left_hand"]), 1.0
    )
    hand_mask.index_fill_(
        0, torch.tensor(SMPLX_object.smplx_mano_vid_dict["right_hand"]), 1.0
    )

    hand_mesh = apply_vertex_mask(hand_mesh, hand_mask) 
    mesh=part_removal(
        mesh,
        hand_mesh,
        0.08,
        device,
        smpl_mesh,
        region="hand"
    )
    hand_mesh.export("./tmp/hand.obj")
    final = poisson(
        [mesh, hand_mesh],
        f"./tmp/final.obj",
        10,
    )

def main(args):
    device = args.device
    root_dir = args.root_dir
    trial_name = args.trial_name
    for subject in tqdm(sorted(os.listdir(root_dir))[14:]):
        print(subject)
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        refine_subject_geometry(subject_dir, trial_name, device)
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认
    parser.add_argument("--root_dir", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set")
    parser.add_argument("--trial_name", type=str, default="nvs_dual_branch@step2_union_v1")
    parser.add_argument("--view_num", type=int, default=20)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    # 修改
    parser.add_argument("--iters", type=int, default=5000, help="optimization step") ##debug
    parser.add_argument('--save_iter', type=int, default=1000)
    args = parser.parse_args()
    
    main(args)