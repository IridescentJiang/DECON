import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
import os
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_nvs import NovelViewPipeline
from torchvision import transforms
import glob
import shutil
from src.utils.util import get_camera
from src.utils.util import (
    save_videos_grid,
    pil_list_to_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference/inference_wild.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3)
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

    stage1_ckpt_dir = config.stage1_ckpt_dir
    stage1_ckpt_step = config.stage1_ckpt_step
    stage2_ckpt_dir = config.stage2_ckpt_dir
    stage2_ckpt_step = config.stage2_ckpt_step
    
    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        config.base_model_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs
        ),
    ).to(dtype=weight_dtype, device="cuda")

    # 定义single和multi两种denoising unet
    denoising_unet_single_view = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs
        ),
    ).to(dtype=weight_dtype, device="cuda") # 这里加载的是SD的unet权重 和我们训练的motion module权重

    denoising_unet_multi_view = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs
        ),
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
    denoising_unet_single_view.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu"
        ),
        # strict=False, # 因为有motion module的参数无法加载，必须不严格
    )
    denoising_unet_multi_view.load_state_dict(
        torch.load(
            os.path.join(stage2_ckpt_dir, f"denoising_unet-{stage2_ckpt_step}.pth"),
            map_location="cpu"
        ),
        # strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu"
        ),
    )
    if normal_guider is not None:
        normal_guider.load_state_dict(
            torch.load(
                os.path.join(stage1_ckpt_dir, f"normal_guider-{stage1_ckpt_step}.pth"),
                map_location="cpu"
            ),
        )
    if depth_guider is not None:
        depth_guider.load_state_dict(
            torch.load(
                os.path.join(stage1_ckpt_dir, f"depth_guider-{stage1_ckpt_step}.pth"),
                map_location="cpu"
            ),
        )
        
    '''下面2个pipe用的都是NovelViewPipeline, 但是传入的denoising_unet不同，一个是single view的，一个是multi view的
    并且attention_mode不同 之后绑定的ref control mode不同
    '''
    # single view attn
    pipe_single_view = NovelViewPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet_single_view,
        normal_guider=normal_guider,
        depth_guider=depth_guider,
        scheduler=scheduler,
        unet_attention_mode="read_single_view",
    )
    pipe_single_view = pipe_single_view.to("cuda", dtype=weight_dtype)

    # multi view attn
    pipe_multi_view = NovelViewPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet_multi_view,
        normal_guider=normal_guider,
        depth_guider=depth_guider,
        scheduler=scheduler,
        unet_attention_mode="read_multi_view",
    )
    pipe_multi_view = pipe_multi_view.to("cuda", dtype=weight_dtype)
    
    vae.eval()
    image_enc.eval()
    reference_unet.eval()
    denoising_unet_single_view.eval()
    denoising_unet_multi_view.eval()
    if normal_guider is not None:
        normal_guider.eval()
    if depth_guider is not None:
        depth_guider.eval()
    
    # person_paths = [
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/1697/render"
    # ]
    validation_dir = config.validation_dir
    # subjects = [
    #     "0000", "0067", "0125", "0161", "0295", "0397", "0400", "0520", "0606", "1122",
    #     "1678", "1854", "1980", "2298", "2410", "1698", "2430"]
    
    subjects = [
        f"{i:04d}" for i in range(0,16)
        ]
    
    for subject in tqdm(subjects):
        # 准备PIL参考帧
        ref_image_pred_path = f"/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/in-the-wild/econ_pymafx/{subject}/000.png"
        ref_image_pred_pil = Image.open(ref_image_pred_path).convert("RGB")
        
        
        # 准备[PIL]normal和depth
        normal_pred_dir = os.path.join(validation_dir, subject, "pymafx/normal")
        depth_pred_dir = os.path.join(validation_dir, subject, "pymafx/depth")

        normal_pred_paths = sorted(glob.glob(os.path.join(normal_pred_dir, "*.png")))
        depth_pred_paths = sorted(glob.glob(os.path.join(depth_pred_dir, "*.png")))
        
        assert len(normal_pred_paths) == len(depth_pred_paths), f"{len(normal_pred_paths) = } != {len(depth_pred_paths) = }"

        video_length = len(normal_pred_paths)
        clip_length = config.clip_length
        step = video_length // clip_length
        batch_index = [i * step for i in range(clip_length)]
        
        normal_pred_list = []
        depth_pred_list = []
        azim_list = []
        elev_list = []
        for index in batch_index:
            normal_pred_path = normal_pred_paths[index]
            normal_pred_pil = Image.open(normal_pred_path).convert("RGB")
            normal_pred_list.append(normal_pred_pil)

            depth_pred_path = depth_pred_paths[index]
            depth_pred_pil = Image.open(depth_pred_path)
            depth_pred_np = np.array(depth_pred_pil)
            depth_pred_np = depth_pred_np / 65535.0
            depth_pred_pil = Image.fromarray(depth_pred_np)
            depth_pred_list.append(depth_pred_pil)
            
            azim = -float(os.path.basename(normal_pred_path).split(".")[0])
            elev = 0.0
            azim_list.append(azim)
            elev_list.append(elev)
            
        camera_list = []
        for azim, elev in zip(azim_list, elev_list):
            camera = get_camera(elev, azim)
            camera_list.append(camera)
        cameras = np.stack(camera_list, axis=0) # (f, 4, 4)
        ref_camera = get_camera(0.0, 0.0)
        
        # pil_list_to_tensor 会把normal, depth, normal_pred, depth_pred, gt转为tensor 便于可视化
        normal_pred_tensor = pil_list_to_tensor(normal_pred_list)
        depth_pred_tensor = pil_list_to_tensor(depth_pred_list).repeat(1, 3, 1, 1, 1)

        
        video_multi_view_wo_smplx = pipe_multi_view(
            ref_image_pred_pil,
            None, # 这里不改None也可以 因为下面scale是0.0 只加权了wo的部分
            None,
            cameras,
            ref_camera,
            width,
            height,
            clip_length,
            args.steps,
            args.cfg,
            generator=generator,
            smplx_guidance_scale=0.0,
        ).videos

        video_multi_view_pred = pipe_multi_view(
            ref_image_pred_pil,
            normal_pred_list,
            depth_pred_list,
            cameras,
            ref_camera,
            width,
            height,
            clip_length,
            args.steps,
            args.cfg,
            generator=generator,
            smplx_guidance_scale=2.0,
        ).videos
        
        
        # 渲染全部视频
        video = torch.cat([ video_multi_view_pred, normal_pred_tensor, (video_multi_view_pred + normal_pred_tensor)/2.0,
                            video_multi_view_wo_smplx, normal_pred_tensor, (video_multi_view_wo_smplx + normal_pred_tensor)/2.0,
                            ], dim=0)

        
        video = video.repeat(1,1,2,1,1) # 沿着时间复制一遍 多转一圈
        print(video.shape)

        out_file = Path(f"{config.output_dir}/{config.exp_name}/inference_wild/{subject}.mp4")
        save_videos_grid(video, out_file, n_rows=3, fps=5)
        
        # copy config.yaml
        tgt_cfg_path = f"{config.output_dir}/{config.exp_name}/inference_wild/inference_wild.yaml"
        if not os.path.exists(tgt_cfg_path):
            shutil.copy(args.config, tgt_cfg_path)
        

        # 只渲染预测视频
        # video = video_pred
        # out_file = Path(f"{config.output_dir}/{config.exp_name}/inference_pred/cfg_{args.cfg}/{ref_name}.mp4")
        # save_videos_grid(video, out_file, n_rows=4, fps=6)  


if __name__ == "__main__":
    main()

    