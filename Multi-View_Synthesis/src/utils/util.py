import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import imageio
from torchvision import transforms


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    
def pil_list_to_tensor(pil_list):
    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )
    tensor_list = []
    for pil in pil_list:
        tensor_list.append(transform(pil))
    return torch.stack(tensor_list, dim=0).transpose(0, 1).unsqueeze(0)


def save_image_seq(video, save_dir):
    # input:
    #     video: torch.Tensor[b c f h w]
    #     save_dir: str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    image_seq = video.squeeze(0).detach().cpu().numpy().transpose(1,2,3,0) # f h w c
    if image_seq.shape[-1]==1:
        image_seq = image_seq[...,0] # 如果单通但的话就变成f h w
    num_frames = image_seq.shape[0]
    angle_step = 360 // num_frames
    for i in range(image_seq.shape[0]):
        image = Image.fromarray((image_seq[i]*255).astype(np.uint8))
        angle = i * angle_step
        save_path = os.path.join(save_dir, f"{angle:03d}.png")
        image.save(save_path)

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

def get_camera(elevation, azimuth):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    
    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    
    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward) # z
    right = np.cross(-up, forward) 
    right /= np.linalg.norm(right) # x
    new_up = np.cross(forward, right) # y
    new_up /= np.linalg.norm(new_up) # y
    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = new_up
    cam2world[:3, 2] = forward
    cam2world[:3, 3] = camera_pos
    return cam2world


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        writer = imageio.get_writer(path, fps=fps)
        for img in pil_images:
            img = np.array(img)
            writer.append_data(img)
        writer.close()
        # codec = "libx264"
        # container = av.open(path, "w")
        # stream = container.add_stream(codec, rate=fps)

        # stream.width = width
        # stream.height = height

        # for pil_image in pil_images:
        #     # pil_image = Image.fromarray(image_arr).convert("RGB")
        #     av_frame = av.VideoFrame.from_image(pil_image)
        #     container.mux(stream.encode(av_frame))
        # container.mux(stream.encode())
        # container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

