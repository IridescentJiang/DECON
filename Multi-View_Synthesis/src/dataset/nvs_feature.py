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


class NvsFeatureDataset(Dataset):
    def __init__(
        self,
        width,
        height,
        n_sample_frames,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
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
        video_meta = self.vid_meta[index]
        rgb_dir = video_meta["rgb"]
        normal_dir = video_meta["normal"]
        depth_dir = video_meta["depth"]
        feat_dir = rgb_dir.replace("render", "feature")
        feat_path = os.path.join(feat_dir, "feature.json")

        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        
        assert len(rgb_paths) == len(normal_paths) == len(depth_paths), f"{len(rgb_paths) = } != {len(normal_paths) = } != {len(depth_paths) = }"
    
        # 准备PIL参考帧
        ref_img_idx = 0 # 正面作为初始帧
        ref_img_path = rgb_paths[ref_img_idx] # 参考帧
        ref_img_pil = Image.open(ref_img_path).convert("RGB")
        
        # 准备[PIL]目标帧、normal、depth，等距在0~71中取
        video_length = len(rgb_paths)
        sample_len = self.n_sample_frames
        step = video_length // sample_len
        batch_index = [i * step for i in range(sample_len)]
        
        vid_pil_image_list = []
        normal_pil_image_list = []
        depth_pil_image_list = []
        for index in batch_index:
            # tgt img
            vid_pil_image = Image.open(rgb_paths[index]).convert("RGB")
            vid_pil_image_list.append(vid_pil_image)
            # normal img
            normal_pil_image = Image.open(normal_paths[index]).convert("RGB")
            normal_pil_image_list.append(normal_pil_image)
            # deoth img 需要转换一下
            depth_pil_image = Image.open(depth_paths[index])
            depth_np_image = np.array(depth_pil_image)
            depth_np_image = depth_np_image / 65535.0
            depth_pil_image = Image.fromarray(depth_np_image)
            depth_pil_image_list.append(depth_pil_image)

        # transform
        state = torch.get_rng_state()
        # state = None
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_normal = self.augmentation(
            normal_pil_image_list, self.cond_transform, state
        )
        pixel_values_depth = self.augmentation(
            depth_pil_image_list, self.cond_transform, state
        ) # f,c,h,w
        pixel_values_ref_img = self.augmentation(ref_img_pil, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        pixel_values_depth = pixel_values_depth.repeat(1, 3, 1, 1) # 3,f,h,w 扩展成3通道

        sample = dict(
            video_dir=rgb_dir,
            pixel_values_vid=pixel_values_vid,
            pixel_values_normal=pixel_values_normal,
            pixel_values_depth=pixel_values_depth,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        
        with open(feat_path, "r") as f:
            feat_dict = json.load(f)

        for key, value in feat_dict.items():
            feat_dict[key] = torch.tensor(value)

        sample.update(feat_dict)

        return sample

    def __len__(self):
        return len(self.vid_meta)
    