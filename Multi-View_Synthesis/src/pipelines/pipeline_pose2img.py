import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.mutual_self_attention import ReferenceAttentionControl
import PIL.Image
import PIL.ImageOps
from PIL import Image

@dataclass
class Pose2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class Pose2ImagePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        normal_guider,
        depth_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        unet_attention_mode="read_concat_attn",
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            normal_guider=normal_guider,
            depth_guider=depth_guider,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.normal_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        self.unet_attention_mode = unet_attention_mode
        ## debug
        # print('!!!!!!!')
        # print(self.unet_attention_mode)


    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(
        self,
        cond_image,
        width,
        height,
        device,
        dtype,
        do_classififer_free_guidance=False,
    ):
        image = self.cond_image_processor.preprocess(
            cond_image, height=height, width=width
        ).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        normal_image,
        depth_image,
        camera,
        ref_camera,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        if self.image_encoder is not None:
            clip_image = self.clip_image_processor.preprocess(
                ref_image.resize((224, 224)), return_tensors="pt"
            ).pixel_values
            uncond_clip_image = torch.zeros_like(clip_image) # 无条件的吧图置0，再送入clip
            
            clip_image_embeds = self.image_encoder(
                clip_image.to(device, dtype=self.image_encoder.dtype)
            ).image_embeds
            uncond_clip_image_embeds = self.image_encoder(
                uncond_clip_image.to(device, dtype=self.image_encoder.dtype)
            ).image_embeds
                
            image_prompt_embeds = clip_image_embeds.unsqueeze(1)
            uncond_image_prompt_embeds = uncond_clip_image_embeds.unsqueeze(1)
            # uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

            if do_classifier_free_guidance:
                image_prompt_embeds = torch.cat(
                    [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
                ) # bs维度拼起来，放在一个batch里面过
            # cfg的时候，clip特征把0和实际特征拼起来，其余所有特征和输入都在bs上repeat2
        else: # 不用clip的时候
            image_prompt_embeds = None
        
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode=self.unet_attention_mode,
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            self.vae.dtype,
            device,
            generator,
        )
        latents = latents.unsqueeze(2)  # (bs, c, 1, h', w')
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare normal & depth condition image
        if self.normal_guider is not None and normal_image is not None:
            normal_cond_tensor = self.normal_image_processor.preprocess(
                normal_image, height=height, width=width
            ) # preprocess出来自动多了b    b c h w
            normal_cond_tensor = normal_cond_tensor.unsqueeze(2)  # (bs, c, f=1, h, w)
            normal_cond_tensor = normal_cond_tensor.to(
                device=device, dtype=self.normal_guider.dtype
            )
            normal_fea = self.normal_guider(normal_cond_tensor)
            normal_fea = (
                torch.cat([normal_fea] * 2) if do_classifier_free_guidance else normal_fea
            ) # 2, 320, 1, 64, 64 这里bs2适用于同时计算con和uncon(cfg没有用pose，这里只是在batch里复制一份用于unc和con计算)
        else:
            normal_fea = None
        
        # depth我们手动处理，不用processor
        # depth_cond_tensor = self.depth_image_processor.preprocess(
        #     depth_image, height=height, width=width
        # ) # bs,1,h,w
        
        if self.depth_guider is not None and depth_image is not None:
            depth_image = depth_image.resize((width, height), resample=PIL.Image.LANCZOS)
            depth_image = [np.array(depth_image).astype(np.float32)]
            depth_image = np.stack(depth_image , axis=0)
            if depth_image.ndim == 3:
                depth_image = depth_image[None, ...]
            depth_cond_tensor = torch.from_numpy(depth_image) # b=1,1,h,w
            depth_cond_tensor = depth_cond_tensor.repeat(1, 3, 1, 1) # bs,3,h,w
            depth_cond_tensor = depth_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
            depth_cond_tensor = depth_cond_tensor.to(
                device=device, dtype=self.depth_guider.dtype
            )
            depth_fea = self.depth_guider(depth_cond_tensor)
            depth_fea = (
                torch.cat([depth_fea] * 2) if do_classifier_free_guidance else depth_fea
            ) # 2, 320, 1, 64, 64 这里bs2适用于同时计算con和uncon
        else:
            depth_fea = None

        # 处理camera
        camera = torch.from_numpy(camera[:3, :3].reshape(-1))[None, None, :].to(device=device, dtype=latents.dtype)  # b=1 f=1 c=9 
        camera = torch.cat([camera] * 2) if do_classifier_free_guidance else camera # b=2 f=1 c=9 如果用cfg

        ref_camera = torch.from_numpy(ref_camera[:3, :3].reshape(-1))[None, :].to(device=device, dtype=latents.dtype)  # b=1 c=9 
        ref_camera = torch.cat([ref_camera] * 2) if do_classifier_free_guidance else ref_camera # b=2 c=9 如果用cfg

        
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t), # t没有做cfg的处理，所以在unet里面会expand处理一下
                        camera=ref_camera,
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )

                    # 2. Update reference unet feature into denosing net 存在denosing unet的bank里
                    reference_control_reader.update(reference_control_writer)
                
                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    camera=camera,
                    normal_cond_fea=normal_fea,
                    depth_cond_fea=depth_fea,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                # 每次单步去噪完就混合cfg的结果，下次去噪再进一步cfg
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0: # false
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear() # 每个样本完全去噪完才清空bank，也就是整个样本推理中只需算一次ref’
            reference_control_writer.clear() # 而训练中每次iter都要换ref图，所以都要清空

        # Post-processing
        image = self.decode_latents(latents)  # (b, c, 1, h, w)

        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)

        if not return_dict:
            return image

        return Pose2ImagePipelineOutput(images=image)

