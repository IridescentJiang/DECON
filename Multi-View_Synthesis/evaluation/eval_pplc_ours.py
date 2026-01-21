import os

import numpy as np
from argparse import ArgumentParser

import torch
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
import lpips

# clip
import clip
from utils.clip_utils import compute_clip_score
from PIL import Image
import cv2

views_dict = {
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],  # ("018", "018")
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    }

def rectify_img(img, center, azim):
    azim_radians = np.radians(azim)
    scale_factor = np.cos(azim_radians)  # 由于是正交投影，缩放因子为cos(azim)
    translation_x = (1 - scale_factor) * center
    M = np.array([
    [scale_factor, 0, translation_x],
    [0, 1, 0]
    ])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img
    
    
def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(1 / mse)
    return psnr


def color_map_forward(rgb):
    dim = rgb.shape[-1]
    if dim==3:
        return rgb.astype(np.float32)/255
    else:
        rgb = rgb.astype(np.float32)/255
        rgb, alpha = rgb[:,:,:3], rgb[:,:,3:]
        rgb = rgb * alpha + (1-alpha)
        return rgb

def main():
    # 只需要路径、实验名（也可以是gt）、和dataset，不用gt，自己和自己测
    parser = ArgumentParser()
    parser.add_argument('--pr',type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set")
    parser.add_argument('--dataset',type=str, default="thuman", help="thuman or customhumans")
    parser.add_argument('--name',type=str, default="gt_gen/nvs_dual_branch@step2_union_v1") # mv下的实验子目录
    parser.add_argument('--step',type=int, default=1) # 计算间隔step的相似度
    args = parser.parse_args()

    pr_dir = args.pr
    lpips_fn = lpips.LPIPS(net='vgg').cuda().eval()
    
    subjects = sorted(os.listdir(pr_dir))
    if args.dataset == 'thuman':
        subjects = [subject for subject in subjects if len(subject)==4]
    if args.dataset == 'customhumans':
        subjects = [subject for subject in subjects if len(subject)>4]
    print(subjects)
    
    pplcs = []
    
    for subject in tqdm(subjects):
        pr_sub_dir = os.path.join(pr_dir, subject, f"mv/{args.name}/rgb")
        
        views = sorted(os.listdir(pr_sub_dir)) # "000.png 018.png ..."
        azim = 360.0/len(views)
        for idx, view in tqdm(enumerate(views)):
            # 读相邻两张图
            idx_next = (idx + args.step) % len(views)
            view_next = views[idx_next]
            img_path = os.path.join(pr_sub_dir, view)
            img_next_path = os.path.join(pr_sub_dir, view_next)
            
            img = imread(img_path)
            img_next = imread(img_next_path)
            if img.shape[:2] != (512, 512):
                img = resize(img, (512,512), preserve_range=True).astype(np.uint8) # pred的全部resize到512比较
            if img_next.shape[:2] != (512, 512):
                img_next = resize(img_next, (512,512), preserve_range=True).astype(np.uint8) # pred的全部resize到512比较
            # 当前图rectify
            # cv2.imwrite('origin.png', img)
            img_rectified = rectify_img(img, center=img.shape[0]/2, azim=azim) # 用间隔角度做校正
            # cv2.imwrite('warp.png', img_rectified)
            
            img_rectified = color_map_forward(img_rectified)
            img_next = color_map_forward(img_next)

            with torch.no_grad():
                img_rectified_tensor = torch.from_numpy(img_rectified.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
                img_next_tensor = torch.from_numpy(img_next.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
                img_rectified_th, img_next_th = img_rectified_tensor*2-1, img_next_tensor*2-1
                score = float(lpips_fn(img_rectified_th, img_next_th).flatten()[0].cpu().numpy())
                score = score / (np.radians(azim)**2)
            pplcs.append(score)

    # msg=f'{args.name}\t{args.dataset}\t{args.mode}\t{np.mean(psnrs):.5f}\t{np.mean(ssims):.5f}\t{np.mean(lpipss):.5f}'
    msg = f'{args.name:<40}\t{args.dataset:<15}\t{args.step:<3}\t{np.mean(pplcs):.5f}'
    print(msg)
    with open('pplc.log','a') as f:
        f.write(msg+'\n')


if __name__=="__main__":
    main()
    
# pose free
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset thuman --name free_gen --step 1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset customhumans --name free_gen -step 1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset thuman --name free_gen --step 2
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset customhumans --name free_gen -step 2

# union v1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v1 --step 1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v1 --step 1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v1 --step 2
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v1 --step 2

# union v2
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v2 --step 1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v2 --step 1
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v2 --step 2
# python eval_pplc_ours.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS/test_set --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v2 --step 2