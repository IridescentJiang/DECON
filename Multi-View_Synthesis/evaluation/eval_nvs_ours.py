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

views_dict = {
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],  # ("018", "018")
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    }

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
    parser = ArgumentParser()
    parser.add_argument('--gt',type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set")
    parser.add_argument('--pr',type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set")
    parser.add_argument('--dataset',type=str, default="thuman", help="thuman or customhumans")
    parser.add_argument('--name',type=str, default="gt_gen/nvs_dual_branch@step2_union_v1") # mv下的实验子目录
    parser.add_argument('--mode',type=str, default="ortho", help="full or ortho")
    args = parser.parse_args()

    gt_dir = args.gt
    pr_dir = args.pr

    lpips_fn = lpips.LPIPS(net='vgg').cuda().eval()
    psnrs, ssims, lpipss, clipss = [], [], [], []
    
    # 准备clip需要的东西
    # clip_model, clip_preprocess = clip.load('ViT-B/32', device="cuda:0")
    clip_model, clip_preprocess = clip.load('ViT-L/14', device="cuda:0")
    views = views_dict[args.mode]
    
    subjects = sorted(os.listdir(pr_dir))
    if args.dataset == 'thuman':
        subjects = [subject for subject in subjects if len(subject)==4]
    if args.dataset == 'customhumans':
        subjects = [subject for subject in subjects if len(subject)>4]
    print(subjects)
    
    for subject in tqdm(subjects):
        pr_sub_dir = os.path.join(pr_dir, subject)
        gt_sub_dir = os.path.join(gt_dir, subject)
        
        for view in tqdm(views):
            pr_view, gt_view = view # "000" "000"
            gt_path = os.path.join(gt_sub_dir, f'mv/gt/rgb/{gt_view}.png')
            pr_path = os.path.join(pr_sub_dir, f'mv/{args.name}/rgb/{pr_view}.png')
            
            # 计算psnr/ssim/lpips
            img_gt_int = imread(gt_path)
            img_pr_int = imread(pr_path)
            # if img_pr_int
            if img_pr_int.shape[:2] != (512, 512):
                img_pr_int = resize(img_pr_int, (512,512), preserve_range=True) # pred的全部resize到512比较

            img_gt = color_map_forward(img_gt_int)
            img_pr = color_map_forward(img_pr_int)
            psnr = compute_psnr_float(img_gt, img_pr)

            with torch.no_grad():
                img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
                img_pr_tensor = torch.from_numpy(img_pr.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
                ssim = float(structural_similarity_index_measure(img_pr_tensor, img_gt_tensor).flatten()[0].cpu().numpy())
                gt_img_th, pr_img_th = img_gt_tensor*2-1, img_pr_tensor*2-1
                score = float(lpips_fn(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())

            # 计算clips
            img_gt = clip_preprocess(Image.open(gt_path)).unsqueeze(0).to("cuda:0")
            img_pr = clip_preprocess(Image.open(pr_path)).unsqueeze(0).to("cuda:0")
            clips = compute_clip_score(img_gt, img_pr, clip_model)
            
            ssims.append(ssim)
            lpipss.append(score)
            psnrs.append(psnr)
            clipss.append(clips)

    # msg=f'{args.name}\t{args.dataset}\t{args.mode}\t{np.mean(psnrs):.5f}\t{np.mean(ssims):.5f}\t{np.mean(lpipss):.5f}'
    msg = f'{args.name:<40}\t{args.dataset:<15}\t{args.mode:<8}\t{np.mean(psnrs):.5f}\t{np.mean(ssims):.5f}\t{np.mean(lpipss):.5f}\t{np.mean(clipss):.5f}'
    print(msg)
    with open('nvs_ours.log','a') as f:
        f.write(msg+'\n')


if __name__=="__main__":
    main()

# union v1
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v1 --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v1 --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v1 --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v1 --mode ortho

# union v2
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v2 --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v2 --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v2 --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v2 --mode ortho
# ------------------------------------------------------------------------------------------------------------------
# w/o smpl
# python eval_nvs_ours.py --dataset thuman --name free_gen --mode full
# python eval_nvs_ours.py --dataset customhumans --name free_gen --mode full
# python eval_nvs_ours.py --dataset thuman --name free_gen --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name free_gen --mode ortho

# w/ optimized_smpl
# python eval_nvs_ours.py --dataset thuman --name optmized_gen --mode full
# python eval_nvs_ours.py --dataset customhumans --name optmized_gen --mode full
# python eval_nvs_ours.py --dataset thuman --name optmized_gen --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name optmized_gen --mode ortho

# w/ predicted_smpl(pixie)
# python eval_nvs_ours.py --dataset thuman --name pixie_gen --mode full
# python eval_nvs_ours.py --dataset customhumans --name pixie_gen --mode full
# python eval_nvs_ours.py --dataset thuman --name pixie_gen --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name pixie_gen --mode ortho

# w/ predicted_smpl(pymafx)
# python eval_nvs_ours.py --dataset thuman --name pymafx_gen --mode full
# python eval_nvs_ours.py --dataset customhumans --name pymafx_gen --mode full
# python eval_nvs_ours.py --dataset thuman --name pymafx_gen --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name pymafx_gen --mode ortho
# ------------------------------------------------------------------------------------------------------------------
# w/o attention
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@stage1 --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@stage1 --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@stage1 --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@stage1 --mode ortho

# w/ temporal attention
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_time --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_time --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_time --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_time --mode ortho

# w/ full attention
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_view --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_view --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_view --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_view --mode ortho

# w/ reverse attention (v3)
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v3 --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v3 --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v3 --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v3 --mode ortho

# w/ normal branch
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_wo_normal --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_wo_normal --mode full
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_wo_normal --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_wo_normal --mode ortho

