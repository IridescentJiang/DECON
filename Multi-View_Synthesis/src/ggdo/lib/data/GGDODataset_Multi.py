import os
import random

import numpy as np
import cv2
import torch
import json

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import apps.generatesmpl_opt_Multi as genSMPL

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

CAMERA_TO_MESH_DISTANCE = 10.0  # This is an arbitrary value set by the rendering script. Can be modified by changing the rendering script.

class SmplOptInferenceDataset(Dataset):

    def __init__(self, opt, validation_mode=False):
        self.opt = opt

        self.validation_mode = validation_mode

        self.is_train = False

        self.depth_map_directory = "rendering_script/inference/trained_depth_maps_refer" # New version (Depth maps trained with only normal - Second Stage maps)

        self.root = "rendering_script/inference/buffer_fixed_full_mesh"

        self.smpl_para_rect = "rendering_script/inference/smpl_para"

        self.img_files = []

        group_name = "multi"
        subject_render_folder = os.path.join(self.root, group_name)
        subject_render_paths_list = [os.path.join(subject_render_folder, f) for f in
                                     os.listdir(subject_render_folder)]
        self.img_files = self.img_files + subject_render_paths_list
        self.img_files = sorted(self.img_files)

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])

        self.to_tensor_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.485, 0.229)
        ])

    def __len__(self):
        return len(self.img_files)

    def get_item(self, index):

        img_path = self.img_files[index]
        id = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        size = (height, width)

        # get subject
        subject = img_path.split('/')[-2]  # e.g. "0507"

        depth_map_path = os.path.join(self.depth_map_directory, subject,
                                      f"{id}" + "_smpl.png")

        # get smpl para rect
        smpl_para_rect_path = os.path.join(self.smpl_para_rect, subject,
                                           f"{id}" + "_smpl_para.json")
        with open(smpl_para_rect_path, 'r') as f:
            data_smpl_rect = json.load(f)

        # 获取 "betas" 和 "poses" 矩阵
        smpl_shapes_rect = torch.tensor(data_smpl_rect['betas'])
        smpl_poses_rect = torch.tensor(data_smpl_rect['poses'])
        smpl_transl_rect = torch.tensor(data_smpl_rect['transl'])
        smpl_transl_rect[1] = -smpl_transl_rect[1]
        smpl_transl_rect[2] = -smpl_transl_rect[2]

        smpl_transl_rect = smpl_transl_rect.unsqueeze(0)
        smpl_poses_rect = torch.cat((smpl_poses_rect, smpl_transl_rect), dim=0)

        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        depth_map = torch.from_numpy(depth_map)
        depth_map = depth_map.unsqueeze(0)

        depth_map_low_res = F.interpolate(torch.unsqueeze(depth_map, 0),
                                          size=genSMPL.calculate_rescaled_size(size))
        depth_map_low_res = depth_map_low_res[0]
        mask_low_pifu = torch.where(depth_map_low_res > 0.5, 1., 0.)

        R = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        calib = torch.eye(4)[:3]

        return {
            'name': id,
            'id': id,
            'mask_low_pifu': mask_low_pifu,
            'smpl_poses_rect': smpl_poses_rect,
            'smpl_shapes_rect': smpl_shapes_rect,
            'rotation_matrix': R,
            'calib': calib,
            'img_size': size
        }

    def __getitem__(self, index):
        return self.get_item(index)