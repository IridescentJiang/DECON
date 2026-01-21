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
    "zero123":{
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],  # ("018", "018")
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    },
    "zero123-xl":{
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    },
    "stable-zero123":{
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    },
    "syncdreamer":{
        "full": [(f"{i:02d}", f"{i:02d}") for i in range(16)],    # ("02", 02"")
        "ortho": [(f"{i:02d}", f"{i:02d}") for i in range(0, 16, 4)],
    },
    "sv3d":{
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],  # ("018", "018")
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    },
    "wonder3d":{
        "full": [('front', "000"), ('front_right', "315"), ("right", "270"), ("back", "180"), ("left", "090"), ("front_left", "045"),],  # ("front_wight")
        "ortho": [('front', "000"), ("right", "270"), ("back", "180"), ("left", "090")],
    },
    "wonder3d_trained":{
        "full": [('front', "000"), ('front_right', "315"), ("right", "270"), ("back", "180"), ("left", "090"), ("front_left", "045"),],  # ("front_wight")
        "ortho": [('front', "000"), ("right", "270"), ("back", "180"), ("left", "090")],
    },
    "champ":{
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],  # ("018", "018")
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    },
    "animate-anyone":{
        "full": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 18)],  # ("018", "018")
        "ortho": [(f"{i:03d}", f"{i:03d}") for i in range(0, 360, 90)],
    },
    
    
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
    parser.add_argument('--gt',type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/test_set")
    parser.add_argument('--pr',type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/sv3d")
    parser.add_argument('--dataset',type=str, default="wild", help="wild or shhq")
    parser.add_argument('--name',type=str, default="sv3d")
    parser.add_argument('--mode',type=str, default="ortho", help="full or ortho")
    args = parser.parse_args()

    gt_dir = args.gt
    pr_dir = args.pr

    lpips_fn = lpips.LPIPS(net='vgg').cuda().eval()
    psnrs, ssims, lpipss, clipss= [], [], [], []
    
    # 准备clip需要的东西
    # clip_model, clip_preprocess = clip.load('ViT-B/32', device="cuda:0")
    clip_model, clip_preprocess = clip.load('ViT-L/14', device="cuda:0")
    views = views_dict[args.name][args.mode]
    
    subjects = sorted(os.listdir(pr_dir))
    if args.dataset == "shhq":
        subjects = [subject for subject in subjects if "shhq" in subject]
    print(subjects)
    
    for subject in tqdm(subjects):
        pr_sub_dir = os.path.join(pr_dir, subject)
        gt_sub_dir = os.path.join(gt_dir, subject)
        
        ## 正面的psnr ssim lpips
        pr_view, gt_view = views[0]
        gt_path = os.path.join(gt_sub_dir, f'ref/rgb.png') # gt用的都是统一的

        if args.name == "wonder3d" or args.name == "wonder3d_trained": # wonder3d要区分rgb和normal
            pr_path = os.path.join(pr_sub_dir, f'gen_rgb/{pr_view}.png')
        else: # 其余正常情况 gen在自己目录下且只有rgb
            pr_path = os.path.join(pr_sub_dir, f'gen/{pr_view}.png')

        img_gt_int = imread(gt_path)
        img_pr_int = imread(pr_path)
        # if img_pr_int
        if img_pr_int.shape[:2] != (512, 512):
            img_pr_int = resize(img_pr_int, (512,512), preserve_range=True) # pred的全部resize到512比较

        img_gt = color_map_forward(img_gt_int)
        img_pr = color_map_forward(img_pr_int)
        psnr = compute_psnr_float(img_gt, img_pr) # psnr
        with torch.no_grad():
            img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            img_pr_tensor = torch.from_numpy(img_pr.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            ssim = float(structural_similarity_index_measure(img_pr_tensor, img_gt_tensor).flatten()[0].cpu().numpy()) # ssim
            gt_img_th, pr_img_th = img_gt_tensor*2-1, img_pr_tensor*2-1
            score = float(lpips_fn(gt_img_th, pr_img_th).flatten()[0].cpu().numpy()) # lpips
        ssims.append(ssim)
        lpipss.append(score)
        psnrs.append(psnr)
        
        ## 其他view的clip (都用正面的gt) 提前读取好
        img_gt = clip_preprocess(Image.open(gt_path)).unsqueeze(0).to("cuda:0")
        for view in tqdm(views):
            pr_view, gt_view = view # "000" "000"
            if args.name == "wonder3d" or args.name == "wonder3d_trained": # wonder3d要区分rgb和normal
                pr_path = os.path.join(pr_sub_dir, f'gen_rgb/{pr_view}.png')
            else: # 其余正常情况 gen在自己目录下且只有rgb
                pr_path = os.path.join(pr_sub_dir, f'gen/{pr_view}.png')
            
            # 计算clips
            img_gt = clip_preprocess(Image.open(gt_path)).unsqueeze(0).to("cuda:0")
            img_pr = clip_preprocess(Image.open(pr_path)).unsqueeze(0).to("cuda:0")
            clips = compute_clip_score(img_gt, img_pr, clip_model)
            clipss.append(clips)

    # msg=f'{args.name}\t{args.dataset}\t{args.mode}\t{np.mean(psnrs):.5f}\t{np.mean(ssims):.5f}\t{np.mean(lpipss):.5f}'
    msg = f'{args.name:<20}\t{args.dataset:<15}\t{args.mode:<10}\t{np.mean(psnrs):.5f}\t{np.mean(ssims):.5f}\t{np.mean(lpipss):.5f}\t{np.mean(clipss):.5f}'
    print(msg)
    with open('nvs.log','a') as f:
        f.write(msg+'\n')


if __name__=="__main__":
    main()

# zero123
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123 --dataset wild --name zero123
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123 --dataset shhq --name zero123

# zero123-xl
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123-xl --dataset wild --name zero123-xl
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123-xl --dataset shhq --name zero123-xl

# sv3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/sv3d --dataset wild --name sv3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/sv3d --dataset shhq --name sv3d

# syncdreamer
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/syncdreamer --dataset wild --name syncdreamer
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/syncdreamer --dataset shhq --name syncdreamer

# wonder3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d --dataset wild --name wonder3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d --dataset shhq --name wonder3d

# wonder3d_trained
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d_trained --dataset wild --name wonder3d_trained
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d_trained --dataset shhq --name wonder3d_trained

# champ
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/champ --dataset wild --name champ
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/champ --dataset shhq --name champ

# animate-anyone
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/animate-anyone --dataset wild --name animate-anyone
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/animate-anyone --dataset shhq --name animate-anyone