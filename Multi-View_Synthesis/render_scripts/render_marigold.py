'''
/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/THuman2/ortho
- scan
    - 0000
        - scan
            - rgb
            - normal
            - depth
            - mask
        - smplx
            - normal
            - depth
            - mask
        - valid 验证渲染结果是否对齐的

        elev.txt
        light.txt
'''
import os

import torch

import numpy as np

import argparse

# add path for demo utils functions
from tqdm import tqdm

import random

from diffusers import MarigoldNormalsPipeline
from PIL import Image

# # 初始化随机数生成器的种子
# random.seed(224)

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

aa_factor = 6

import numpy as np


def render_marigold_normal(rgb_pil, mask_pil, marigold_dir, angle, method="marigold", device=device):
    if method == "marigold":
        pipe = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v0-1",
            variant="fp16",
            torch_dtype=torch.float16
        ).to(device)
        normal_np = pipe(rgb_pil, num_inference_steps=25).prediction
        mask_np = np.array(mask_pil)[None, :, :, None]

        def normalize_normal_map(normal_np):
            norms = np.linalg.norm(normal_np, axis=-1, keepdims=True)
            normal_np = normal_np / norms
            normal_np = (normal_np + 1.0) / 2.0
            return normal_np

        # normalize & mask bg
        normal_np = normalize_normal_map(normal_np)
        normal_np = normal_np * (mask_np > 0)
        normal_pil = Image.fromarray((normal_np[0] * 255).astype(np.uint8)).convert("RGB")

        del pipe
        torch.cuda.empty_cache()

        save_path = os.path.join(marigold_dir, f"{angle:03d}.png")
        normal_pil.save(save_path)

        return normal_pil

    else:
        raise NotImplementedError


def read_rgb_mask(scan_rgb_dir, scan_mask_dir, angle):
    # 构建文件路径，假设文件名格式与之前保存一致
    rgb_path = os.path.join(scan_rgb_dir, f"{angle:03d}.png")
    mask_path = os.path.join(scan_mask_dir, f"{angle:03d}.png")

    # 使用PIL读取图像
    rgb_pil = Image.open(rgb_path).convert('RGB')
    mask_pil = Image.open(mask_path).convert('L')  # 将蒙版加载为灰度图

    # 可能需要对 mask 进行处理以确保格式一致性
    # mask_pil = mask_pil.convert('1')  # 如果是二值图，可以选择转换为'1'模式

    return rgb_pil, mask_pil


def render_thuman2(start, end, interval):
    img_dir = f"/home/lab/multiHumanRecon/Dataset/data_rendering/THuman2/ortho_{interval}"

    for subject in tqdm(range(start, end)):
        print("render", subject, "start...")

        subject_dir = f"{img_dir}/{subject:04d}"

        scan_dir = f"{subject_dir}/scan"

        scan_rgb_dir = f"{scan_dir}/rgb"
        scan_mask_dir = f"{scan_dir}/mask"
        scan_marigold_dir = f"{scan_dir}/marigold"

        os.makedirs(scan_marigold_dir, exist_ok=True)


        for angle in tqdm(range(0, 360, interval)):
            scan_rgb, scan_mask = read_rgb_mask(scan_rgb_dir, scan_mask_dir, angle)

            scan_marigold = render_marigold_normal(scan_rgb, scan_mask, scan_marigold_dir, angle, method="marigold", device=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--interval", type=int, default=5)

    args = parser.parse_args()
    # 初始化随机数生成器的种子
    random.seed(args.begin)
    render_thuman2(args.begin, args.end, args.interval)  # 正面为000的渲染
