import json
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import glob
import os
from src.utils.util import get_camera


class MultiViewDataset(Dataset):
    def __init__(
        self,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(1.0, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
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
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        human_meta = self.vid_meta[index]
        
        # sample = dict(
        #     video_dir=rgb_dir,
        #     pixel_values_vid=pixel_values_vid,
        #     pixel_values_normal=pixel_values_normal,
        #     pixel_values_depth=pixel_values_depth,
        #     pixel_values_ref_img=pixel_values_ref_img,
        #     clip_ref_img=clip_ref_img,
        #     tgt_camera=camera,
        #     ref_camera=ref_camera,
        # )
        # 需要的数据:
        #         gt: scan_rgb scan_normal
        #         smplx: smplx_semantic smplx_normal
        #         ref: scan_rgb marigold_normal
        
        # gt        
        scan_rgb_dir = human_meta["scan"]["rgb"]
        scan_normal_dir = human_meta["scan"]["normal"]
        scan_rgb_paths = sorted(glob.glob(os.path.join(scan_rgb_dir, "*.png")))
        scan_normal_paths = sorted(glob.glob(os.path.join(scan_normal_dir, "*.png")))
        # smplx
        smplx_semantic_dir = human_meta["smplx"]["semantic"]
        smplx_normal_dir = human_meta["smplx"]["normal"]
        smplx_semantic_paths = sorted(glob.glob(os.path.join(smplx_semantic_dir, "*.png")))
        smplx_normal_paths = sorted(glob.glob(os.path.join(smplx_normal_dir, "*.png")))
        # ref
        ref_normal_dir = human_meta["ref"]["normal"] # 用marigold
        ref_normal_paths = sorted(glob.glob(os.path.join(ref_normal_dir, "*.png")))
        
        assert len(scan_rgb_paths) == len(scan_normal_paths) \
                == len(smplx_semantic_paths) == len(smplx_normal_paths)
                # == len(ref_normal_paths) ref normal没有全部渲染!
        
        # ref rgb/normal camera
        ref_img_idx = random.choice([0, 1, 2, len(scan_rgb_paths)-1, len(scan_rgb_paths)-2]) # 正面±小范围作为初始帧
        ref_rgb_path = scan_rgb_paths[ref_img_idx] # ref_rgb用的是gt
        ref_normal_path = next(path for path in ref_normal_paths if os.path.basename(path) == os.path.basename(ref_rgb_path))
                                        # ref_normal用的是marigold估计的 BUG:ref_normal不全 不能用index 用和rgb_path相同basename的
        ref_rgb_pil = Image.open(ref_rgb_path).convert("RGB")
        ref_normal_pil = Image.open(ref_normal_path).convert("RGB")
        ref_azim = -float(os.path.basename(ref_rgb_path).split('.')[0]) # 参考帧的旋转角
        ref_elev = 0.0 # 暂时默认0 elev
        
        # cond semantic/normal    tgt rgb/normal 等距在0~71中取
        total_length = len(scan_rgb_paths)
        sample_len = self.n_sample_frames
        step = total_length // sample_len
        batch_index = [(ref_img_idx + i * step) % total_length for i in range(sample_len)]
        
        cond_semantic_pil_list = []
        cond_normal_pil_list = []
        
        tgt_rgb_pil_list = []
        tgt_normal_pil_list = []
        
        tgt_azim_list = []
        tgt_elev_list = []
        
        for index in batch_index:
            # cond semantic/normal
            cond_semantic_pil_list.append(Image.open(smplx_semantic_paths[index]).convert("RGB"))
            cond_normal_pil_list.append(Image.open(smplx_normal_paths[index]).convert("RGB"))
            # tgt rgb/normal
            tgt_rgb_pil_list.append(Image.open(scan_rgb_paths[index]).convert("RGB"))
            tgt_normal_pil_list.append(Image.open(scan_normal_paths[index]).convert("RGB"))            
            # tgt azim和elev
            tgt_azim = -float(os.path.basename(scan_rgb_paths[index]).split('.')[0]) # 目标帧的旋转角
            tgt_elev = 0.0 # 暂时默认0 elev
            tgt_azim_list.append(tgt_azim)
            tgt_elev_list.append(tgt_elev)
        
        # camera
        camera_list = [] # [np(4,4)]
        for tgt_azim, tgt_elev in zip(tgt_azim_list, tgt_elev_list):
            azim = tgt_azim - ref_azim
            elev = tgt_elev - ref_elev
            camera = get_camera(elev, azim)
            camera_list.append(camera)
        camera = np.stack(camera_list, axis=0) # (f,4,4)
        # 多返回一个参考帧的camera参数，目前只有一帧，且认为是正面帧(azim=0,elev=0)，之后还会引入其他帧
        ref_camera = get_camera(0.0, 0.0)
        
        # transform
        state = torch.get_rng_state()
        # ref [-1,1]
        ref_rgb_img = self.augmentation(ref_rgb_pil, self.pixel_transform, state) # c h w
        ref_normal_img = self.augmentation(ref_normal_pil, self.pixel_transform, state)
        # cond semantic/normal [0,1]
        cond_semantic_imgs = self.augmentation(cond_semantic_pil_list, self.cond_transform, state)
        cond_normal_imgs = self.augmentation(cond_normal_pil_list, self.cond_transform, state)
        # tgt rgb/normal (vae) [-1,1]
        tgt_rgb_imgs = self.augmentation(tgt_rgb_pil_list, self.pixel_transform, state)
        tgt_normal_imgs = self.augmentation(tgt_normal_pil_list, self.pixel_transform, state)
        # clip image 用rgb
        clip_image = self.clip_image_processor(images=ref_rgb_pil, return_tensors="pt").pixel_values[0]

        sample = dict(
            person_dir="/".join(scan_rgb_dir.split('/')[:-2]), # 路径
            # ref rgb/normal 一张图
            ref_rgb_img=ref_rgb_img,
            ref_normal_img=ref_normal_img,
            # cond semantic/normal 多张图
            cond_semantic_imgs=cond_semantic_imgs,
            cond_normal_imgs=cond_normal_imgs,
            # tgt rgb/normal 多张图
            tgt_rgb_imgs=tgt_rgb_imgs,
            tgt_normal_imgs=tgt_normal_imgs,
            # clip image
            clip_image=clip_image,
            # camera
            tgt_camera=camera,
            ref_camera=ref_camera,
        )
        # dataset返回的都是预处理过的tensor，并且repeat过，可以直接送入unet的
        
        return sample

    def __len__(self):
        return len(self.vid_meta)
    

