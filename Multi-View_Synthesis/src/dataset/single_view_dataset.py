import json
import random

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import os
import glob
import numpy as np
from src.utils.util import get_camera

# def get_camera(elevation, azimuth):
#     elevation = np.radians(elevation)
#     azimuth = np.radians(azimuth)
#     # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
#     x = np.cos(elevation) * np.sin(azimuth)
#     y = np.sin(elevation)
#     z = np.cos(elevation) * np.cos(azimuth)
    
#     # Calculate camera position, target, and up vectors
#     camera_pos = np.array([x, y, z])
#     target = np.array([0, 0, 0])
#     up = np.array([0, 1, 0])
    
#     # Construct view matrix
#     forward = target - camera_pos
#     forward /= np.linalg.norm(forward)
#     right = np.cross(forward, up)
#     right /= np.linalg.norm(right)
#     new_up = np.cross(right, forward)
#     new_up /= np.linalg.norm(new_up)
#     cam2world = np.eye(4)
#     cam2world[:3, :3] = np.array([right, new_up, -forward]).T
#     cam2world[:3, 3] = camera_pos
#     return cam2world

def normalize_depth_pil(depth_pil):
    # PIL [0,65535] -> PIL [0,1]
    depth_np = np.array(depth_pil)
    depth_np = depth_np / 65535.0
    depth_pil = Image.fromarray(depth_np)
    return depth_pil

class SingleViewDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        # img_ratio=(0.9, 1.0),
        img_ratio=(1.0, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/thuman2_meta.json"],
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        # -----
        # vid_meta format:
        # [{'rgb': , 'normal': , 'depth':},
        #  {'rgb': , 'normal': , 'depth':},]
        # -----
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor() # 应该只是clip前的预处理

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        human_meta = self.vid_meta[index]
        
        # 需要的数据:
        #     gt: scan_rgb scan_normal (smplx_depth)
        #     smplx: smplx_semantic smplx_normal
        #     ref: scan_rgb marigold_normal
        ## TODO: 如果需要ref depth和gt depth的话,可以再加
        # gt        
        # print(human_meta["scan"])
        scan_rgb_dir = human_meta["scan"]["rgb"]
        scan_normal_dir = human_meta["scan"]["normal"]
        scan_rgb_paths = sorted(glob.glob(os.path.join(scan_rgb_dir, "*.png")))
        scan_normal_paths = sorted(glob.glob(os.path.join(scan_normal_dir, "*.png")))
        # smplx
        smplx_semantic_dir = human_meta["smplx"]["semantic"]
        # smplx_semantic_dir = human_meta["smplx"]["normal"]
        smplx_normal_dir = human_meta["smplx"]["normal"]
        smplx_depth_dir = human_meta["smplx"]["depth"] # Optional
        smplx_semantic_paths = sorted(glob.glob(os.path.join(smplx_semantic_dir, "*.png")))
        smplx_normal_paths = sorted(glob.glob(os.path.join(smplx_normal_dir, "*.png")))
        smplx_depth_paths = sorted(glob.glob(os.path.join(smplx_depth_dir, "*.png")))
        # ref
        ref_normal_dir = human_meta["ref"]["normal"] # 用marigold
        ref_normal_paths = sorted(glob.glob(os.path.join(ref_normal_dir, "*.png")))
        

        assert len(scan_rgb_paths) == len(scan_normal_paths) \
                == len(smplx_semantic_paths) == len(smplx_normal_paths) == len(smplx_depth_paths)
                # == len(ref_normal_paths) ref normal没有全部渲染!
                     
        ref_img_idx = random.choice([0, 1, 2, len(scan_rgb_paths)-1, len(scan_rgb_paths)-2]) # 正面作为初始帧
        tgt_img_idx = random.randint(0, len(scan_rgb_paths) - 1) # 随机选一帧做目标

        # ref
        ref_rgb_path = scan_rgb_paths[ref_img_idx] # ref_rgb用的是gt
        ref_normal_path = next(path for path in ref_normal_paths if os.path.basename(path) == os.path.basename(ref_rgb_path))
                                        # ref_normal用的是marigold估计的 BUG:ref_normal不全 不能用index 用和rgb_path相同basename的
        # cond (smplx)
        cond_semantic_path = smplx_semantic_paths[tgt_img_idx]
        cond_normal_path = smplx_normal_paths[tgt_img_idx]
        cond_depth_path = smplx_depth_paths[tgt_img_idx]
        # tgt (gt)
        tgt_rgb_path = scan_rgb_paths[tgt_img_idx] # 目标帧
        tgt_normal_path = scan_normal_paths[tgt_img_idx] # 目标normal
        
        ## camera 要计算tgt相对ref的elev和azim 我们标号的时候，是人体逆时针转，所以相当于相机顺时针转！也就是相机角度应该是负的
        ref_azim = -float(os.path.basename(ref_rgb_path).split('.')[0]) # 参考帧的旋转角
        ref_elev = 0.0 # 暂时默认0 elev
        tgt_azim = -float(os.path.basename(tgt_rgb_path).split('.')[0]) # 目标帧的旋转角
        tgt_elev = 0.0 # 暂时默认0 elev
        
        # 相对
        azim = tgt_azim - ref_azim # 不用放到360也行应该 因为后面算camera参数负角度也一样
        elev = tgt_elev - ref_elev
        
        # 计算camera的R矩阵
        camera = get_camera(elev, azim) # 这里把4*4都返回 最后在训练时候决定是否用
        # 多返回一个参考帧的camera参数，目前只有一帧，且认为是正面帧(azim=0,elev=0)，之后还会引入其他帧
        ref_camera = get_camera(0.0, 0.0)
        
        # 加载pil
        ref_rgb_pil = Image.open(ref_rgb_path).convert("RGB")
        ref_normal_pil = Image.open(ref_normal_path).convert("RGB")
        
        cond_semantic_pil = Image.open(cond_semantic_path).convert("RGB")
        cond_normal_pil = Image.open(cond_normal_path).convert("RGB")
        cond_depth_pil = normalize_depth_pil(Image.open(cond_depth_path)) # depth被归一化到[0,1]
        
        tgt_rgb_pil = Image.open(tgt_rgb_path).convert("RGB")
        tgt_normal_pil = Image.open(tgt_normal_path).convert("RGB")
        

        state = torch.get_rng_state()
        # ref rgb/normal (vae) [-1,1]
        ref_rgb_img = self.augmentation(ref_rgb_pil, self.transform, state)
        ref_normal_img = self.augmentation(ref_normal_pil, self.transform, state)
        # cond semantic/normal/depth [0,1] 
        cond_semantic_img = self.augmentation(cond_semantic_pil, self.cond_transform, state)
        cond_normal_img = self.augmentation(cond_normal_pil, self.cond_transform, state)
        cond_depth_img = self.augmentation(cond_depth_pil, self.cond_transform, state).repeat(3, 1, 1) # 3通道复制
        # tgt rgb/normal (vae) [-1,1]
        tgt_rgb_img = self.augmentation(tgt_rgb_pil, self.transform, state)
        tgt_normal_img = self.augmentation(tgt_normal_pil, self.transform, state)
        # 最后返回一个ref rgb的clip image (其实用不到 branch合并后也不知道应该用哪个)
        clip_image = self.clip_image_processor(images=ref_rgb_pil, return_tensors="pt" ).pixel_values[0]
        
        
        sample = dict(
            person_dir="/".join(scan_rgb_dir.split('/')[:-2]), # 路径
            # ref rgb/normal
            ref_rgb_img=ref_rgb_img, # 参考rgb
            ref_normal_img=ref_normal_img, # 参考normal
            # cond semantic/normal/depth
            cond_semantic_img=cond_semantic_img, # smplx语义
            cond_normal_img=cond_normal_img, # smplx normal
            cond_depth_img=cond_depth_img, # smplx depth
            # tgt rgb/normal
            tgt_rgb_img=tgt_rgb_img, # 目标rgb
            tgt_normal_img=tgt_normal_img, # 目标normal
            # clip rgb
            clip_image=clip_image, # clip图像 其实就是参考图像预处理 224x224
            # camera
            tgt_camera=camera, # 相机参数
            ref_camera=ref_camera,
        )
        
        # # 画图确认
        # import torchvision.transforms.functional as TF
        # tensor_to_image = TF.to_pil_image
        # tgt_img_s = tensor_to_image((tgt_img+1)/2)
        # tgt_normal_img_s = tensor_to_image(tgt_normal_img)
        # tgt_depth_img_s = tensor_to_image(tgt_depth_img)
        # ref_img_vae_s = tensor_to_image((ref_img_vae+1)/2)
        # clip_image_s = tensor_to_image((clip_image - clip_image.min())/(clip_image.max() - clip_image.min()))
        
        # tgt_img_s.save(f"img/tgt_img.png")
        # tgt_normal_img_s.save(f"img/tgt_normal_img.png")
        # tgt_depth_img_s.save(f"img/tgt_depth_img.png")
        # ref_img_vae_s.save(f"img/ref_img_vae.png")
        # clip_image_s.save(f"img/clip_image.png")

        return sample

    def __len__(self):
        return len(self.vid_meta)
