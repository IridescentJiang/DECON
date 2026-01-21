'''只在定义的时候写一下depth_guider
包括depth_guider定义/Net(log_validation传过去的net里也有)
另外dataset和训练数据预处理的时候处理一下depth
其余forward全部不带depth了!
'''
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
# from tqdm.auto import tqdm
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from diffusers import StableDiffusionPipeline
from src.dataset.single_view_dataset import SingleViewDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.pipelines.pipeline_nvs import NovelViewPipeline
from src.utils.util import (
    delete_additional_ckpt, 
    import_filename, 
    seed_everything, 
    get_camera, 
    pil_list_to_tensor,
    save_videos_grid,
)
import random
import shutil
import glob
warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# rgb_mean = 0.484496
# rgb_std = 1.228611
# normal_mean = -0.219865
# normal_std = 1.443801

rgb_mean = 0.484496
rgb_std = 1.229314
normal_mean = -0.219865
normal_std = 1.445059

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        semantic_guider: PoseGuider,
        normal_guider: PoseGuider,
        depth_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.semantic_guider = semantic_guider
        self.normal_guider = normal_guider
        self.depth_guider = depth_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        # 加噪输入
        noisy_rgb_latents,
        noisy_normal_latents,
        # 噪声步
        timesteps,
        # 相机
        cameras, 
        ref_cameras,
        # 参考
        ref_rgb_latents,
        ref_normal_latents,
        image_prompt_embeds,  # clip
        # cond smplx
        cond_semantic_img,
        cond_normal_img,
        # 是否ref img的cfg
        uncond_fwd: bool = False,
    ):
        if self.semantic_guider is not None and cond_semantic_img is not None: # normal guider和normal_img都不为None（normal_img为None表示cfg drop）
            semantic_fea = self.semantic_guider(cond_semantic_img.to(device="cuda")) # normal_guider和tensor都是float32，出来的fea是float16
        else:
            semantic_fea = None
        if self.normal_guider is not None and cond_normal_img is not None:
            normal_fea = self.normal_guider(cond_normal_img.to(device="cuda"))
        else:
            normal_fea = None
        # if self.depth_guider is not None and cond_depth_img is not None:
        #     depth_fea = self.depth_guider(cond_depth_img.to(device="cuda"))
        # else:
        #     depth_fea = None
        

        # cfg是对clip和ref同时做，在这里是对ref做，main里面已经对clip做了
        # 以及，cfg是对整个batch做
        if not uncond_fwd: # 有条件的时候走这里 更新ref feature map（clip都是None了）
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                # ref输入
                sample=ref_rgb_latents, sample_noisy_list=[ref_normal_latents],
                timestep=ref_timesteps,
                encoder_hidden_states=image_prompt_embeds,
                camera=ref_cameras, # ref用的camera是0的
                return_dict=False,
            ) # reference_control_writer里面包含了reference_unet，因此记录了reference_unet前传过程中的一些feat_map
            self.reference_control_reader.update(self.reference_control_writer)

        # 如果无条件的话，reference_unet不用前传
        # 并且，reference_control_writer和reference_control_reader的bank都是空的
        # 所以denoising_unet的的basicTrans拼了空的，也就是自己本身不变，做self-att
        model_pred = self.denoising_unet(
            # 加噪输入
            sample=noisy_rgb_latents, sample_noisy_list=[noisy_normal_latents],
            timestep=timesteps,
            encoder_hidden_states=image_prompt_embeds,
            camera=cameras,
            # cond smplx
            pose_cond_fea=semantic_fea,
            pose_cond_fea_list=[normal_fea], # 这里暂时直接不放入depth. 对于[normal_fea]在ucon的时候是[None]
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    unet_attention_mode,
    clip_length=24,
):
    logger.info("Running validation... ")
    net.eval()

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    semantic_guider = ori_net.semantic_guider
    normal_guider = ori_net.normal_guider

    generator = torch.manual_seed(42)
    
    # ##debug
    # vae = vae.to(dtype=torch.float32)
    # if image_enc is not None:
    #     image_enc = image_enc.to(dtype=torch.float32)

    pipe = NovelViewPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        semantic_guider=semantic_guider,
        normal_guider=normal_guider,
        scheduler=scheduler,
        unet_attention_mode=unet_attention_mode,
    )
    pipe = pipe.to(accelerator.device)
    
    # subjects = ["0080", "0125", "1680", "2438", "2440"]
    subjects = ["0099"]
    validation_dir = "/root/autodl-tmp/dataset/THuman2_val/ortho_5" ##debug

    results = []
    for subject in subjects:
        subject_dir = os.path.join(validation_dir, subject)
        # 准备ref rgb/normal
        ref_rgb_path = os.path.join(subject_dir, "scan/rgb/000.png")
        ref_normal_path = os.path.join(subject_dir, "marigold/normal/000.png")
        ref_rgb_pil = Image.open(ref_rgb_path).convert("RGB")
        ref_normal_pil = Image.open(ref_normal_path).convert("RGB")
        
        # 准备cond semantic/normal
        cond_semantic_dir = os.path.join(subject_dir, "smplx/semantic")
        cond_normal_dir = os.path.join(subject_dir, "smplx/normal")
        cond_semantic_paths = sorted(glob.glob(os.path.join(cond_semantic_dir, "*.png")))
        cond_normal_paths = sorted(glob.glob(os.path.join(cond_normal_dir, "*.png")))
        
        # 准备cond_pred semantic/normal (pymafx)
        # cond_pred_semantic_dir = os.path.join(subject_dir, "pymafx/semantic")
        # cond_pred_normal_dir = os.path.join(subject_dir, "pymafx/normal")
        cond_pred_semantic_dir = os.path.join(subject_dir, "smplx/semantic")
        cond_pred_normal_dir = os.path.join(subject_dir, "smplx/normal")
        cond_pred_semantic_paths = sorted(glob.glob(os.path.join(cond_pred_semantic_dir, "*.png")))
        cond_pred_normal_paths = sorted(glob.glob(os.path.join(cond_pred_normal_dir, "*.png")))
        
        # 准备gt rgb/normal 用于可视化
        gt_rgb_paths = sorted(glob.glob(os.path.join(subject_dir, "scan/rgb/*.png")))
        gt_normal_paths = sorted(glob.glob(os.path.join(subject_dir, "scan/normal/*.png")))
        
        assert len(cond_semantic_paths) == len(cond_normal_paths), f"{len(cond_semantic_paths) = } != {len(cond_normal_paths) = }"

        total_length = len(cond_semantic_paths) # 一个路径下的图片总数 72/60 要间隔取
        step = total_length // clip_length
        batch_index = [i * step for i in range(clip_length)]
        
        cond_semantic_list = []
        cond_normal_list = []
        cond_pred_semantic_list = []
        cond_pred_normal_list = []
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
            # semantic pred
            cond_pred_semantic_path = cond_pred_semantic_paths[index]
            cond_pred_semantic_pil = Image.open(cond_pred_semantic_path).convert("RGB")
            cond_pred_semantic_list.append(cond_pred_semantic_pil)
            # normal pred
            cond_pred_normal_path = cond_pred_normal_paths[index]
            cond_pred_normal_pil = Image.open(cond_pred_normal_path).convert("RGB")
            cond_pred_normal_list.append(cond_pred_normal_pil)
            
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
        cond_pred_semantic_tensor = pil_list_to_tensor(cond_pred_semantic_list)
        cond_normal_tensor = pil_list_to_tensor(cond_normal_list) # b=1 c f h w
        cond_pred_normal_tensor = pil_list_to_tensor(cond_pred_normal_list)
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
            num_inference_steps=20,
            guidance_scale=2.5,
            smplx_guidance_scale=2.5,
            guidance_rescale=0.7,
            generator=generator,
            )
        rgb_video = output.rgb_videos
        normal_video = output.normal_videos
        
        # 用pred的前传处理 
        output_pred = pipe(
            # ref rgb/normal
            ref_rgb_pil,
            ref_normal_pil,
            # cond semantic/normal
            cond_pred_semantic_list,
            cond_pred_normal_list,
            camera,
            ref_camera,
            width,
            height,
            clip_length,
            num_inference_steps=20,
            guidance_scale=2.5,
            smplx_guidance_scale=2.5,
            guidance_rescale=0.7,
            generator=generator,
            )
        rgb_video_pred = output_pred.rgb_videos
        normal_video_pred = output_pred.normal_videos
        
        # 无smplx前传处理 
        output_wo_smplx = pipe(
            # ref rgb/normal
            ref_rgb_pil,
            ref_normal_pil,
            # cond semantic/normal
            None,
            None,
            camera,
            ref_camera,
            width,
            height,
            clip_length,
            num_inference_steps=20,
            guidance_scale=2.5,
            smplx_guidance_scale=0,
            guidance_rescale=0.7,
            generator=generator,
            )
        rgb_video_wo_smplx = output_wo_smplx.rgb_videos
        normal_video_wo_smplx = output_wo_smplx.normal_videos

        video = torch.cat([
            rgb_video, normal_video, cond_semantic_tensor, cond_normal_tensor,
            rgb_video_pred, normal_video_pred, cond_pred_semantic_tensor, cond_pred_normal_tensor,
            rgb_video_wo_smplx, normal_video_wo_smplx, gt_rgb_tensor, gt_normal_tensor
        ], dim=0)
        
        results.append({"name": f"{subject}", "vid": video})

        ##debug
        # break
    # vae = vae.to(dtype=torch.float16)
    # image_enc = image_enc.to(dtype=torch.float16)

    del pipe
    torch.cuda.empty_cache()
    net.train()

    return results


def main(cfg, cfg_path):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # kwargs_2 = InitProcessGroupKwargs(timeout=timedelta(seconds=60), backend="gloo")
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    output_dir = cfg.output_dir
    save_dir = f"{output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if accelerator.is_main_process:
        shutil.copy(cfg_path, f"{save_dir}/config.yaml")

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs) 
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    # '''    
    # 定义ddim策略（diffuers库）
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    # 定义vae（diffusers库），加载权重
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    
    ## 定义clip图像编码器（transformer库），加载权重
    if cfg.use_clip_cross_attention:
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path,
        ).to(dtype=weight_dtype, device="cuda")
    else:
        image_enc = None
    
    ## 定义reference net（自己定义的unet2d），加载权重
    cfg.unet_additional_kwargs.update({"use_clip_cross_attention": cfg.use_clip_cross_attention}) # 把是否用clip也传给unet参数
    reference_unet = UNet2DConditionModel.from_pretrained_2d( # 这里的from_pretrained_2d也是自己定义的，为了用attn1初始化attn0，以及缺少camera的初始化
        cfg.base_model_path,
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs
        ), # from_petrained中可以接受多余的参数并且筛除掉
    ).to(device="cuda")
    
    ## 定义unet（自己定义的unet3d），加载权重
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs
        ),
    ).to(device="cuda")
    
    ## 定义各个pose guider
    # 先加载预训练权重
    if cfg.pose_guider_pretrain:
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
    
    # 定义semantic_guider
    if cfg.use_semantic_guider:
        if cfg.pose_guider_pretrain:
            semantic_guider = PoseGuider(**cfg.pose_guider_kwargs).to(device="cuda")
            # load pretrained controlnet-openpose params for pose_guider
            miss_normal, _ = semantic_guider.load_state_dict(state_dict_to_load, strict=False)
            logger.info(f"Missing key for normal guider: {len(miss_normal)}")
        else:
            semantic_guider = PoseGuider(conditioning_embedding_channels=320).to(device="cuda")
    else:
        semantic_guider = None
    
    # 定义normal_guider
    if cfg.use_normal_guider:
        if cfg.pose_guider_pretrain:
            normal_guider = PoseGuider(**cfg.pose_guider_kwargs).to(device="cuda")
            # load pretrained controlnet-openpose params for pose_guider
            miss_normal, _ = normal_guider.load_state_dict(state_dict_to_load, strict=False)
            logger.info(f"Missing key for normal guider: {len(miss_normal)}")
        else:
            normal_guider = PoseGuider(conditioning_embedding_channels=320).to(device="cuda")
    else:
        normal_guider = None
    
    # 定义depth_guider
    if cfg.use_depth_guider:
        if cfg.pose_guider_pretrain:
            depth_guider = PoseGuider(**cfg.pose_guider_kwargs).to(device="cuda")
            # load pretrained controlnet-openpose params for pose_guider
            miss_depth, _ = depth_guider.load_state_dict(state_dict_to_load, strict=False)
            logger.info(f"Missing key for depth guider: {len(miss_depth)}")
        else:
            depth_guider = PoseGuider(conditioning_embedding_channels=320).to(device="cuda")
    else:
        depth_guider = None
    
    # Freeze
    vae.requires_grad_(False)
    if image_enc is not None:
        image_enc.requires_grad_(False)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    reference_unet.requires_grad_(True)

    if semantic_guider is not None:
        semantic_guider.requires_grad_(True)
    if normal_guider is not None:
        normal_guider.requires_grad_(True)
    if depth_guider is not None:
        depth_guider.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode = cfg.unet_attention_mode, # ["read_concat_attn", "read_cross_attn"]
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        semantic_guider,
        normal_guider,
        depth_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention: # true
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing: # false
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else: # false
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else: # false
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = SingleViewDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        img_scale=(1.0, 1.0),
        data_meta_paths=cfg.data.meta_paths,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=1 # 4
    )  # debug
    print("【Dataloader Length】")
    print(len(train_dataloader))

    # Prepare everything with our `accelerator`.
    # vae和img_enc不用过吗？
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            f"{os.path.basename(output_dir)}:{exp_name}",
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    net.train()
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save 这里是直接读取accelerate存的整个state，而不是部分ckpt
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            ##debug
            # 一个iter
            with accelerator.accumulate(net):
                ## 处理tgt rgb/normal (过vae、加噪、优化目标,作为输入和输出)
                # 过vae
                tgt_rgb_img = batch["tgt_rgb_img"].to(weight_dtype)
                tgt_normal_img = batch["tgt_normal_img"].to(weight_dtype)
                with torch.no_grad():
                    tgt_rgb_latents = vae.encode(tgt_rgb_img).latent_dist.sample() # 512 -> 64
                    tgt_rgb_latents = tgt_rgb_latents.unsqueeze(2) # (b, c, 1, h, w)
                    tgt_rgb_latents = tgt_rgb_latents * 0.18215
                    
                    tgt_normal_latents = vae.encode(tgt_normal_img).latent_dist.sample()
                    tgt_normal_latents = tgt_normal_latents.unsqueeze(2)
                    tgt_normal_latents = tgt_normal_latents * 0.18215
                    # TODO: tgt归一化分布
                    tgt_normal_latents = (
                        (tgt_normal_latents - normal_mean) / normal_std * rgb_std + rgb_mean
                    )
                # 采样不同噪声
                rgb_noise = torch.randn_like(tgt_rgb_latents)
                normal_noise = torch.randn_like(tgt_normal_latents)
                if cfg.noise_offset > 0.0:
                    rgb_noise += cfg.noise_offset * torch.randn(
                        (rgb_noise.shape[0], rgb_noise.shape[1], 1, 1, 1),
                        device=rgb_noise.device,) # 不同bc加不一样的offset，但是像素点之间offset相同
                    normal_noise += cfg.noise_offset * torch.randn(
                        (normal_noise.shape[0], normal_noise.shape[1], 1, 1, 1),
                        device=normal_noise.device,)
                assert tgt_rgb_latents.shape == tgt_normal_latents.shape
                bsz = tgt_rgb_latents.shape[0]
                
                # rgb/normal共享timestep
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=tgt_rgb_latents.device,
                )
                timesteps = timesteps.long()
                
                # 加噪
                noisy_rgb_latents = train_noise_scheduler.add_noise(
                    tgt_rgb_latents, rgb_noise, timesteps
                )
                noisy_normal_latents = train_noise_scheduler.add_noise(
                    tgt_normal_latents, normal_noise, timesteps
                ) # (b, c, 1, h, w) # BUG:检查
                
                ## 准备去噪目标(v_prediction的时候和noise+无噪的latents都有关系)
                if train_noise_scheduler.prediction_type == "epsilon":
                    rgb_target = rgb_noise
                    normal_target = normal_noise
                    target = torch.cat([rgb_target, normal_target], dim=1) # Unet估计的目标是把rgb和normal的噪声在C维度拼在一起
                elif train_noise_scheduler.prediction_type == "v_prediction": #这里
                    rgb_target = train_noise_scheduler.get_velocity(
                        tgt_rgb_latents, rgb_noise, timesteps
                    )
                    normal_target = train_noise_scheduler.get_velocity(
                        tgt_normal_latents, normal_noise, timesteps
                    )
                    target = torch.cat([rgb_target, normal_target], dim=1) # Unet估计的目标是把rgb和normal的噪声在C维度拼在一起
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                
                ## 准备camera
                cameras = batch["tgt_camera"].to(weight_dtype) # bs, 4, 4
                if cfg.camera_use_radius:
                    cameras = cameras.reshape(bsz, -1).unsqueeze(1) # bs f=1 16
                else: # 走这里
                    cameras = cameras[:, :3, :3].reshape(bsz, -1).unsqueeze(1)  # bs f=1 9
                # ref camera
                ref_cameras = batch["ref_camera"].to(weight_dtype) # bs, 4, 4
                if cfg.camera_use_radius:
                    ref_cameras = ref_cameras.reshape(bsz, -1) # bs 16
                else: # 走这里
                    ref_cameras = ref_cameras[:, :3, :3].reshape(bsz, -1) # bs 9 
                    # 目前认为每个batch内只给一个参考图，而且是要送给2Dunet的，所以ref camera暂时没有f维度
                
                ## 准备smplx条件cond
                drop_smplx = random.random() < cfg.drop_smplx_ratio # 以一定的比例drop smplx
                # drop_smplx = True ##debug
                if not drop_smplx:
                    cond_semantic_img = batch["cond_semantic_img"].unsqueeze(2) # (bs, 3, 1, 512, 512)
                    cond_normal_img = batch["cond_normal_img"].unsqueeze(2) # (bs, 3, 1, 512, 512)
                    cond_depth_img = batch["cond_depth_img"].unsqueeze(2) # (bs, 3, 1, 512, 512)                    
                else:
                    cond_semantic_img = None
                    cond_normal_img = None
                    cond_depth_img = None

                ## 准备ref rgb/normal 包括vae和clip
                uncond_fwd = random.random() < cfg.uncond_ratio        
                # uncond_fwd = True ##debug
                clip_image_list = []
                ref_rgb_list = []
                ref_normal_list = []
                for batch_idx, (ref_rgb_img, ref_normal_img, clip_img) in enumerate(
                    zip(
                        batch["ref_rgb_img"],
                        batch["ref_normal_img"],
                        batch["clip_image"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img)) # clip img置0
                    else:
                        clip_image_list.append(clip_img)
                    ref_rgb_list.append(ref_rgb_img) # 不管哪种cfg都要加入正常的ref_img （在net里再考虑是否update用于cfg）
                    ref_normal_list.append(ref_normal_img)
                
                with torch.no_grad():
                    ref_rgb = torch.stack(ref_rgb_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_rgb_latents = vae.encode(ref_rgb).latent_dist.sample()  # (bs, d, 64, 64) d=4
                    ref_rgb_latents = ref_rgb_latents * 0.18215
                    
                    ref_normal = torch.stack(ref_normal_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_normal_latents = vae.encode(ref_normal).latent_dist.sample()  # (bs, d, 64, 64) d=4
                    ref_normal_latents = ref_normal_latents * 0.18215
                    # TODO: tgt归一化分布
                    ref_normal_latents = (
                        (ref_normal_latents - normal_mean) / normal_std * rgb_std + rgb_mean
                    )
                    
                    if image_enc is not None: 
                        clip_img = torch.stack(clip_image_list, dim=0).to(
                            dtype=image_enc.dtype, device=image_enc.device
                        )
                        clip_image_embeds = image_enc(
                            clip_img.to("cuda", dtype=weight_dtype)
                        ).image_embeds # (bs, 768)
                        clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                        image_prompt_embeds = clip_image_embeds
                    else: # 如果不用clip的话 clip emb是None
                        image_prompt_embeds = None
                
                # print("begin forward!")
                model_pred = net(
                    # 加噪输入
                    noisy_rgb_latents,
                    noisy_normal_latents,
                    # 噪声步
                    timesteps,
                    # 相机
                    cameras, 
                    ref_cameras,
                    # 参考
                    ref_rgb_latents,
                    ref_normal_latents,
                    image_prompt_embeds,  # clip
                    # cond smplx
                    cond_semantic_img,
                    cond_normal_img,
                    # 是否ref img的cfg
                    uncond_fwd,
                ) # b 2c=8 f 64 64
                
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # BUG: 已修复 v_prediction的snr权重 
                    snr = compute_snr(train_noise_scheduler, timesteps) # shape B batch内用的time不一样 所以snr也不一样
                    mse_loss_weights = torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    # 对于batch内每一个 取snr和snr_gamma中小的 最后那个取[0]是value和index中取前者
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1) # min{snr, snr_gamma}/(snr+1)
                    elif train_noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr # min{snr, snr_gamma}/snr
                    else:
                        raise ValueError(f"Unknown prediction type {train_noise_scheduler.config.prediction_type}")
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    ) # b 2c f h w
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    ) # 除了b维度都先取mean 再在b上乘上各自噪声水平对应的权重 (把2c也取mean了 相当于把rgb和branch取mean)
                    loss = loss.mean() # 全部取mean

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # # # Backpropagate
                ##debug
                # print("No backward!!!")
                # accelerator.backward(loss)
                accelerator.backward(loss, retain_graph=True)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            ##debug

            accelerator.wait_for_everyone()
            
            if accelerator.sync_gradients:
                reference_control_reader.clear() # 每次iter都要把reader和writer的bank清空
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0: # 间隔2000step的时候存整个state
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        delete_additional_ckpt(save_dir, 1)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            unet_attention_mode=cfg.unet_attention_mode,
                            clip_length=cfg.data.n_sample_frames,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            with TemporaryDirectory() as temp_dir:
                                out_file = Path(
                                    f"{temp_dir}/{global_step:06d}-{sample_name}.mp4"
                                )
                                print(out_file)
                                save_videos_grid(vid, out_file, n_rows=4, fps=6)
                                mlflow.log_artifact(out_file)

                        
                        # # log完之后，调用下面的函数，把其中basicTrans的前传重新绑定到o_classifier_free_guidance=False的状态
                        # # 同时里面也清空了bank
                        reference_control_reader.register_reference_hooks(
                            mode=cfg.unet_attention_mode,
                            do_classifier_free_guidance=False, # 重点是这里
                            # reference_attn=True, # 默认True
                        )
                        
                        reference_control_writer.register_reference_hooks(
                            mode="write",
                            do_classifier_free_guidance=False, # 重点是这里
                            # reference_attn=True,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            
            # print("iter done!")

            if global_step >= cfg.solver.max_train_steps:
                break
 
        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=1,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=1,
            )
            if semantic_guider is not None:
                save_checkpoint(
                    unwrap_net.semantic_guider,
                    save_dir,
                    "semantic_guider",
                    global_step,
                    total_limit=1,
                )
            if normal_guider is not None:
                save_checkpoint(
                    unwrap_net.normal_guider,
                    save_dir,
                    "normal_guider",
                    global_step,
                    total_limit=1,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config, args.config)
    
    
