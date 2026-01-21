import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

def plot_kps(img, kps, idx):
    # img: C H W tensor
    # kps: N 2 tensor
    img = img.detach().cpu()
    kps = kps.detach().cpu()
    img_np = np.array(to_pil_image(img))
    kps_np = kps.detach().numpy() * np.array([img_np.shape[1], img_np.shape[0]])
    plt.imshow(img_np)
    plt.scatter(kps_np[:, 0], kps_np[:, 1], c='r', s=5)
    for i, (x, y) in enumerate(kps_np):
        plt.text(x+1, y+1, str(i), color='g', fontsize=10)
    plt.savefig(f'vis/debug_{idx:02d}.png')
    plt.close()
    