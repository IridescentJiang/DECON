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
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from torchvision import transforms
import glob
from src.utils.util import (
    save_videos_grid,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference/inference_stage_2_rpe.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args


def pil_list_to_tensor(pil_list):
    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )
    tensor_list = []
    for pil in pil_list:
        tensor_list.append(transform(pil))
    return torch.stack(tensor_list, dim=0).transpose(0, 1).unsqueeze(0)

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
    
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        os.path.join(stage2_ckpt_dir, f"motion_module-{stage2_ckpt_step}.pth"),
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs
        ),
        load_position_encoding_weights=True,
    ).to(dtype=weight_dtype, device="cuda") # 这里加载的是SD的unet权重 和我们训练的motion module权重


    # 没有时间模块的denoising unet
    denoising_unet_wo_mm = UNet3DConditionModel.from_pretrained_2d(
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
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu"
        ),
        strict=False, # 因为有motion module的参数无法加载，必须不严格
    )
    denoising_unet_wo_mm.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu"
        ),
        strict=False,
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

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        normal_guider=normal_guider,
        depth_guider=depth_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    # 没有时间模块的pipe
    pipe_wo_mm = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet_wo_mm,
        normal_guider=normal_guider,
        depth_guider=depth_guider,
        scheduler=scheduler,
    )
    pipe_wo_mm = pipe_wo_mm.to("cuda", dtype=weight_dtype)
    
    vae.eval()
    image_enc.eval()
    reference_unet.eval()
    denoising_unet.eval()
    denoising_unet_wo_mm.eval()
    if normal_guider is not None:
        normal_guider.eval()
    if depth_guider is not None:
        depth_guider.eval()
    
    # person_paths = [
    #     "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/1697/render"
    # ]

    person_paths = [
        "/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/{:04d}/render".format(i) for i in range(2400, 2445)
    ]
    person_paths.append("/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2300/render")
    person_paths.append("/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/2350/render")
    person_paths.append("/apdcephfs_cq10/share_1330077/hexu/data/THuman2.1/img/1697/render")
    
    for person_path in tqdm(person_paths):
        # 准备PIL参考帧
        ref_image_path = os.path.join(person_path, "000.png")
        ref_name = ref_image_path.split("/")[-3]
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        
        # 准备[PIL]normal和depth
        normal_dir = person_path.replace("render", "normal")
        depth_dir = person_path.replace("render", "depth")
        normal_pred_dir = person_path.replace("render", "normal_estimate")
        depth_pred_dir = person_path.replace("render", "depth_estimate")

        normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        normal_pred_paths = sorted(glob.glob(os.path.join(normal_pred_dir, "*.png")))
        depth_pred_paths = sorted(glob.glob(os.path.join(depth_pred_dir, "*.png")))

        # 准备gt 用于可视化
        gt_paths = sorted(glob.glob(os.path.join(person_path, "*.png")))
        
        assert len(normal_paths) == len(depth_paths), f"{len(normal_paths) = } != {len(depth_paths) = }"

        video_length = len(normal_paths)
        clip_length = config.clip_length
        step = video_length // clip_length
        batch_index = [i * step for i in range(clip_length)]
        
        normal_list = []
        depth_list = []
        normal_pred_list = []
        depth_pred_list = []
        gt_list = [] # gt用于可视化

        for index in batch_index:
            normal_path = normal_paths[index]
            normal_pil = Image.open(normal_path).convert("RGB")
            normal_list.append(normal_pil)
            
            depth_path = depth_paths[index]
            depth_pil = Image.open(depth_path)
            depth_np = np.array(depth_pil)
            depth_np = depth_np / 65535.0
            depth_pil = Image.fromarray(depth_np)
            depth_list.append(depth_pil)

            normal_pred_path = normal_pred_paths[index]
            normal_pred_pil = Image.open(normal_pred_path).convert("RGB")
            normal_pred_list.append(normal_pred_pil)

            depth_pred_path = depth_pred_paths[index]
            depth_pred_pil = Image.open(depth_pred_path)
            depth_pred_np = np.array(depth_pred_pil)
            depth_pred_np = depth_pred_np / 65535.0
            depth_pred_pil = Image.fromarray(depth_pred_np)
            depth_pred_list.append(depth_pred_pil)
            
            # gt用于可视化
            gt_path = gt_paths[index]
            gt_pil = Image.open(gt_path).convert("RGB")
            gt_list.append(gt_pil)
            

        # pil_list_to_tensor 会把normal, depth, normal_pred, depth_pred, gt转为tensor 便于可视化
        normal_tensor = pil_list_to_tensor(normal_list) # b=1 c f h w
        normal_pred_tensor = pil_list_to_tensor(normal_pred_list)
        depth_tensor = pil_list_to_tensor(depth_list).repeat(1, 3, 1, 1, 1)
        depth_pred_tensor = pil_list_to_tensor(depth_pred_list).repeat(1, 3, 1, 1, 1)
        gt_tensor = pil_list_to_tensor(gt_list)

        
        # 前传处理(包括两种normal)
        pipeline_output = pipe(
            ref_image_pil,
            normal_list,
            depth_list,
            width,
            height,
            clip_length,
            args.steps,
            args.cfg,
            generator=generator,
        )
        video = pipeline_output.videos

        pipeline_output_pred = pipe(
            ref_image_pil,
            normal_pred_list,
            depth_pred_list,
            width,
            height,
            clip_length,
            args.steps,
            args.cfg,
            generator=generator,
        )
        video_pred = pipeline_output_pred.videos

        # 不加时间模块的前传处理（包括两种normal）
        pipe_wo_mm_output = pipe_wo_mm(
            ref_image_pil,
            normal_list,
            depth_list,
            width,
            height,
            clip_length,
            args.steps,
            args.cfg,
            generator=generator,
        )
        video_wo_mm = pipe_wo_mm_output.videos

        pipe_wo_mm_output_pred = pipe_wo_mm(
            ref_image_pil,
            normal_pred_list,
            depth_pred_list,
            width,
            height,
            clip_length,
            args.steps,
            args.cfg,
            generator=generator,
        )
        video_wo_mm_pred = pipe_wo_mm_output_pred.videos
        
        # 渲染全部视频
        video = torch.cat([video, video_wo_mm, normal_tensor, depth_tensor, # 真实normal的有无时间模块
                           video_pred, video_wo_mm_pred, normal_pred_tensor, depth_pred_tensor, # 预测normal的有无时间模块
                           gt_tensor], dim=0)
        
        video = video.repeat(1,1,2,1,1) # 沿着时间复制一遍 多转一圈
        print(video.shape)

        out_file = Path(f"{config.output_dir}/{config.exp_name}/inference/cfg_{args.cfg}/{ref_name}.mp4")
        save_videos_grid(video, out_file, n_rows=4, fps=6)

        # 只渲染预测视频
        # video = video_pred
        # out_file = Path(f"{config.output_dir}/{config.exp_name}/inference_pred/cfg_{args.cfg}/{ref_name}.mp4")
        # save_videos_grid(video, out_file, n_rows=4, fps=6)  


if __name__ == "__main__":
    main()

    