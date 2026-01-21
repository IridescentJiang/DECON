import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from src.dataset.single_sparse_dataset import NvsDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything
import random
import cv2
import imageio
import glob
from torchvision import transforms
from src.utils.util import (
    save_videos_grid,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference/inference_stage_1.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args



def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")
    
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device="cuda")

    if config.use_normal_guider:
        normal_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device="cuda"
        )
    else:
        normal_guider = None

    if config.use_depth_guider:
        depth_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device="cuda"
        )
    else:
        depth_guider = None

    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        # strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    if normal_guider is not None:
        normal_guider.load_state_dict(
            torch.load(config.normal_guider_path, map_location="cpu"),
        )
    if depth_guider is not None:
        depth_guider.load_state_dict(
            torch.load(config.depth_guider_path, map_location="cpu"),
        )

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        normal_guider=normal_guider,
        depth_guider=depth_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)
    
    vae.eval()
    image_enc.eval()
    reference_unet.eval()
    denoising_unet.eval()
    if normal_guider is not None:
        normal_guider.eval()
    if depth_guider is not None:
        depth_guider.eval()
    
    # person_paths = [
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2300/render", # 见过的
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2350/render", # 见过的
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2421/render", # 没见过的
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2430/render", # 没见过的
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2440/render", # 没见过的
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2410/render", # 没见过的
    # ]

    person_paths = [
        "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/{:04d}/render".format(i) for i in range(2395, 2445)
    ]
    person_paths.append("/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2300/render")
    person_paths.append("/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2350/render")
    

    angles = ["{:03d}.png".format(i) for i in range(0, 360, 5)]

    for person_path in person_paths:
        # 准备PIL参考帧
        ref_image_path = os.path.join(person_path, "000.png")
        ref_name = ref_image_path.split("/")[-3]
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        
        # 准备[PIL]normal和depth
        normal_dir = person_path.replace("render", "normal")
        depth_dir = person_path.replace("render", "depth")
        normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        
        # 准备gt 用于可视化
        gt_paths = sorted(glob.glob(os.path.join(person_path, "*.png")))
        
        assert len(normal_paths) == len(depth_paths), f"{len(normal_paths) = } != {len(depth_paths) = }"

        video_length = len(normal_paths)
        clip_length = config.clip_length
        step = video_length // clip_length
        batch_index = [i * step for i in range(clip_length)]
        
        normal_list = []
        gt_list = []
        image_list = []
        
        for idx in batch_index:
            normal_image_path = normal_paths[idx]
            depth_image_path = depth_paths[idx]
            gt_image_path = gt_paths[idx]

            normal_image_pil = Image.open(normal_image_path).convert("RGB")
            normal_list.append(normal_image_pil)
            
            depth_image_pil = Image.open(depth_image_path)
            depth_image_np = np.array(depth_image_pil)
            depth_image_np = depth_image_np / 65535.0
            depth_image_pil = Image.fromarray(depth_image_np)
            # depth_list.append(depth_image_pil)
            
            gt_image_pil = Image.open(gt_image_path).convert("RGB")
            gt_list.append(gt_image_pil)

            image = pipe(
                ref_image_pil,
                normal_image_pil,
                depth_image_pil,
                width,
                height,
                30,
                3.5,
                generator=generator,
            ).images
            image_list.append(image)
        
        # 处理完一个人的，保存一下
        # 把normal图转为tensor，便于可视化
        normal_tensor_list = []
        normal_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        for normal_image_pil in normal_list:
            normal_tensor_list.append(normal_transform(normal_image_pil)) 
        normal_tensor = torch.stack(normal_tensor_list, dim=0)  # (f, c, h, w)
        normal_tensor = normal_tensor.transpose(0, 1) # c, f, h, w
        
        # 把gt图转为tensor，便于可视化
        gt_tensor_list = []
        gt_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        for gt_image_pil in gt_list:
            gt_tensor_list.append(gt_transform(gt_image_pil))
        gt_tensor = torch.stack(gt_tensor_list, dim=0)
        gt_tensor = gt_tensor.transpose(0, 1)
        
        
        video = torch.cat(image_list, dim=2) # 时间维度cat
        normal_tensor = normal_tensor.unsqueeze(0) # b=1 c f h w
        gt_tensor = gt_tensor.unsqueeze(0) # gt
        video = torch.cat([video, normal_tensor, gt_tensor], dim=0)
        
        out_file = Path(f"{config.output_dir}/{config.exp_name}/inference/{ref_name}.mp4")
        save_videos_grid(video, out_file, n_rows=3, fps=6)



if __name__ == "__main__":
    main()

    