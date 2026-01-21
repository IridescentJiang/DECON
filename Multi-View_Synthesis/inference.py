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
    save_image_seq,
)


def parse_args():
    parser = argparse.ArgumentParser()
    ## 只需要改下面3就行
    # 直接用实验路径下转移的yaml
    parser.add_argument("--config", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/nvs_dual_branch/step2_union_v2/config.yaml") 
    # 第2阶段的ckpt step
    parser.add_argument("--stage2_ckpt_step", type=int, default=None)
    # smplx的类型
    parser.add_argument("--cond_smpl_type", type=str, choices=["gt", "pymafx"], default="gt")
    # 测试目录
    parser.add_argument("--test_dir", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set")
    
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--cfg", type=float, default=2.5) # 2.5
    parser.add_argument("--smplx_guidance_scale", type=float, default=3.0) # 3.0
    parser.add_argument("--guidance_rescale", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=25) # 27
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    
    ## ckpt路径
    stage1_ckpt_dir = config.stage1_ckpt_dir
    stage1_ckpt_step = config.stage1_ckpt_step
    
    stage2_ckpt_dir = os.path.dirname(args.config)
    if args.stage2_ckpt_step is not None:
        stage2_ckpt_step = args.stage2_ckpt_step
    else: # 没指定的话自动读取最新的
        checkpoints = os.listdir(stage2_ckpt_dir)
        checkpoints = [d for d in checkpoints if d.startswith("denoising_unet")]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )
        latest_ckpt = checkpoints[-1]
        stage2_ckpt_step = int(latest_ckpt.split("-")[1].split(".")[0])
    config.update(
        {
        "stage2_ckpt_step": stage2_ckpt_step,
        "stage2_ckpt_dir": stage2_ckpt_dir}
    )
    print(f"Use stage1: {stage1_ckpt_dir}, step: {stage1_ckpt_step}")
    print(f"Use stage2: {stage2_ckpt_dir}, step: {stage2_ckpt_step}")
    
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    ## 定义vae和clip
    vae = AutoencoderKL.from_pretrained(
        config.vae_model_path,
    ).to("cuda", dtype=weight_dtype)

    if config.use_clip_cross_attention:
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")
    else:
        image_enc = None
    
    ## 定义两个unet 需要补充信息
    config.unet_additional_kwargs.update({"use_clip_cross_attention": config.use_clip_cross_attention}) 
    if (
        hasattr(config, "unet_additional_kwargs") and
        hasattr(config.unet_additional_kwargs, "motion_module_kwargs") and
        hasattr(config.unet_additional_kwargs.motion_module_kwargs, "temporal_position_encoding_type") and
        config.unet_additional_kwargs.motion_module_kwargs.temporal_position_encoding_type == "RPE"
        ):
        config.unet_additional_kwargs.motion_module_kwargs.update(
            {"temporal_position_encoding_max_len": config.data.n_sample_frames}
            ) # 序列长用于确认rpe长度
    
    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        config.base_model_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs
        ),
    ).to(dtype=weight_dtype, device="cuda")

    # BUG: 因为会用master初始化branch并检查存在性,所以这里必须根据是否使用mm给mm_path
    if config.unet_additional_kwargs.use_motion_module:
        mm_path = config.mm_path
    else:
        mm_path = "" # 不用mm的时候 不会加载mm的预训练权重
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        mm_path, 
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs
        ),
    ).to(dtype=weight_dtype, device="cuda") # 这里加载的是SD的unet权重 和我们训练的motion module权重


    ## 定义guider
    if config.use_semantic_guider:
        semantic_guider = PoseGuider(**config.pose_guider_kwargs).to(device="cuda")
    else:
        semantic_guider = None
    if config.use_normal_guider:
        normal_guider = PoseGuider(**config.pose_guider_kwargs).to(device="cuda")
    else:
        normal_guider = None
    
    ## 定义噪声策略
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        ) # BUG: 注意beta_scale 训练采用scaled_linear 推理用默认的linear 之前的inference的yml都写错了!
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    ## 加载ckpt
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage2_ckpt_dir, f"denoising_unet-{stage2_ckpt_step}.pth"), # 去噪用stage2 其余都用stage1
            map_location="cpu"
        ), # 推理阶段肯定要strict
        # strict=False if config.unet_additional_kwargs.use_motion_module else True, # 有motion module参数无法加载的话不严格
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu"
        ),
    )
    if semantic_guider is not None:
        semantic_guider.load_state_dict(
            torch.load(
                os.path.join(stage1_ckpt_dir, f"semantic_guider-{stage1_ckpt_step}.pth"),
                map_location="cpu",
            ),
            # strict=False,
        )
    if normal_guider is not None:
        normal_guider.load_state_dict(
            torch.load(
                os.path.join(stage1_ckpt_dir, f"normal_guider-{stage1_ckpt_step}.pth"),
                map_location="cpu",
            ),
            # strict=False,
        )
        
    ## pipe
    pipe = NovelViewPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        semantic_guider=semantic_guider,
        normal_guider=normal_guider,
        scheduler=scheduler,
        unet_attention_mode=config.unet_attention_mode,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    
    vae.eval()
    if image_enc is not None:
        image_enc.eval()
    reference_unet.eval()
    denoising_unet.eval()
    if semantic_guider is not None:
        semantic_guider.eval()
    if normal_guider is not None:
        normal_guider.eval()
    
    
    ## Inference begin!
    test_dir = args.test_dir
    subjects = sorted(os.listdir(test_dir))
    # subjects = subjects[6:]
    clip_length = config.data.n_sample_frames
    
    for subject in subjects:
        # if subject != "2430":
        #     continue
        print(f"【Generation】{subject} begin...")
        subject_dir = os.path.join(test_dir, subject)
        # 准备ref rgb/normal
        ref_rgb_path = os.path.join(subject_dir, "ref/rgb.png")
        ref_normal_path = os.path.join(subject_dir, "ref/normal.png")
        ref_rgb_pil = Image.open(ref_rgb_path).convert("RGB")
        ref_normal_pil = Image.open(ref_normal_path).convert("RGB")
        
        # 准备cond semantic/normal
        cond_semantic_dir = os.path.join(subject_dir, "cond", args.cond_smpl_type, "semantic")
        cond_normal_dir = os.path.join(subject_dir, "cond", args.cond_smpl_type, "normal")
                                    # 例如test_set/cond/gt/normal
        cond_semantic_paths = sorted(glob.glob(os.path.join(cond_semantic_dir, "*.png")))
        cond_normal_paths = sorted(glob.glob(os.path.join(cond_normal_dir, "*.png")))
        
        # 准备gt rgb/normal 用于可视化
        gt_rgb_paths = sorted(glob.glob(os.path.join(subject_dir, "mv/gt/rgb/*.png")))
        gt_normal_paths = sorted(glob.glob(os.path.join(subject_dir, "mv/gt/normal/*.png")))
        
        assert len(cond_semantic_paths) == len(cond_normal_paths), f"{len(cond_semantic_paths) = } != {len(cond_normal_paths) = }"

        total_length = len(cond_semantic_paths) # 一个路径下的图片总数 72/60 要间隔取
        step = total_length // clip_length
        batch_index = [i * step for i in range(clip_length)]
        
        cond_semantic_list = []
        cond_normal_list = []
        gt_rgb_list = [] # gt用于可视化
        gt_normal_list = []
        azim_list = []
        elev_list = []
        for index in batch_index:
            # semantic
            cond_semantic_path = cond_semantic_paths[index]
            cond_semantic_pil = Image.open(cond_semantic_path).convert("RGB")
            cond_semantic_list.append(cond_semantic_pil)
            # normal
            cond_normal_path = cond_normal_paths[index]
            cond_normal_pil = Image.open(cond_normal_path).convert("RGB")
            cond_normal_list.append(cond_normal_pil)
            
            # gt rgb/normal用于可视化
            gt_rgb_path = gt_rgb_paths[index]
            gt_rgb_pil = Image.open(gt_rgb_path).convert("RGB")
            gt_rgb_list.append(gt_rgb_pil)
            gt_normal_path = gt_normal_paths[index]
            gt_normal_pil = Image.open(gt_normal_path).convert("RGB")
            gt_normal_list.append(gt_normal_pil)
            
            azim = -float(os.path.basename(gt_rgb_path).split(".")[0])
            elev = 0.0
            azim_list.append(azim)
            elev_list.append(elev)

        camera_list = []
        for azim, elev in zip(azim_list, elev_list):
            camera = get_camera(elev, azim)
            camera_list.append(camera)
        camera = np.stack(camera_list, axis=0) # (f, 4, 4)
        ref_camera = get_camera(0.0, 0.0)

        # pil_list_to_tensor 会把normal, depth, normal_pred, depth_pred, gt转为tensor 便于可视化
        cond_semantic_tensor = pil_list_to_tensor(cond_semantic_list) # b=1 c f h w
        cond_normal_tensor = pil_list_to_tensor(cond_normal_list) # b=1 c f h w
        gt_rgb_tensor = pil_list_to_tensor(gt_rgb_list)
        gt_normal_tensor = pil_list_to_tensor(gt_normal_list)
        
        # debug
        # 前传处理 
        output = pipe(
            # ref rgb/normal
            ref_rgb_pil,
            ref_normal_pil,
            # cond semantic/normal
            cond_semantic_list,
            cond_normal_list,
            camera,
            ref_camera,
            width,
            height,
            clip_length,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            smplx_guidance_scale=args.smplx_guidance_scale,
            guidance_rescale=args.guidance_rescale,
            generator=generator,
            )
        rgb_video = output.rgb_videos # b=1 c f h w
        normal_video = output.normal_videos

        video = torch.cat([
            rgb_video, gt_rgb_tensor, cond_semantic_tensor,
            normal_video, gt_normal_tensor, cond_normal_tensor,
        ], dim=0)
        # video = torch.cat([
        #     rgb_video, normal_video, 
        #     cond_semantic_tensor, cond_normal_tensor,
        #     gt_rgb_tensor, gt_normal_tensor
        # ], dim=0)
        
        video = video.repeat(1,1,2,1,1) # 沿着时间复制一遍 多转一圈
        ## 视频存在实验路径下
        out_file = Path(f"{os.path.dirname(args.config)}/inference/{subject}.mp4") 
        save_videos_grid(video, out_file, n_rows=3, fps=5)
        
        ## 图片存在test_set下
        exp_name_prefix = args.config.split("/")[-3] # conf路径的倒数3 如nvs_dual_branch
        exp_name_postfix = args.config.split("/")[-2] # conf路径的倒数2 如step2_union_v1
        exp_name = f"{exp_name_prefix}@{exp_name_postfix}"
        dir_to_save_img = f"{subject_dir}/mv/gt_gen/{exp_name}"
        if not os.path.exists(dir_to_save_img):
            os.makedirs(dir_to_save_img, exist_ok=True)
            
        save_image_seq(rgb_video, os.path.join(dir_to_save_img, "rgb")) 
        save_image_seq(normal_video, os.path.join(dir_to_save_img, "normal"))
        
        ## cfg也拷贝一份到test里面
        args_conf = OmegaConf.create(vars(args))
        conf = OmegaConf.merge(args_conf, config)
        tgt_cfg_path = f"{dir_to_save_img}/nvs.yaml"
        OmegaConf.save(conf, tgt_cfg_path)


if __name__ == "__main__":
    main()

    