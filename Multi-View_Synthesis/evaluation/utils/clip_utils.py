"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


@torch.no_grad()
def compute_clip_score(img_gt, img_pr, model):
    # logit_scale = model.logit_scale.exp() # 100
    real_features = model.encode_image(img_gt)
    fake_features = model.encode_image(img_pr)

    # normalize features
    real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
    fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
    
    # score = logit_scale * (fake_features * real_features).sum()
    score = (fake_features * real_features).sum()
    score = torch.maximum(score, torch.tensor(0.0))
    return score.detach().cpu().numpy()