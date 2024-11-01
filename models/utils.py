'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-01 16:18:17
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-01 16:18:37

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import logging
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class AffinePatch(nn.Module):

    def __init__(self, ):
        super(AffinePatch, self).__init__()

    def forward(self, img, pred, pad_width=0):
        cos_thetas = torch.unsqueeze(torch.unsqueeze(pred[:, 0], dim=1), dim=2)
        sin_thetas = torch.unsqueeze(torch.unsqueeze(pred[:, 1], dim=1), dim=2)
        tx = (-torch.unsqueeze(torch.unsqueeze(pred[:, 2], dim=1), dim=2) * 2 /
              (2 * pad_width + img.shape[3]))
        ty = (-torch.unsqueeze(torch.unsqueeze(pred[:, 3], dim=1), dim=2) * 2 /
              (2 * pad_width + img.shape[2]))
        ones_v = torch.ones_like(tx)
        zeros_v = torch.zeros_like(tx)

        img_pad = F.pad(
            img,
            (pad_width, pad_width, pad_width, pad_width, 0, 0, 0, 0),
            mode="constant",
            value=0.0,
        )

        As = torch.cat(
            [
                torch.cat([cos_thetas, -sin_thetas, zeros_v], dim=2),
                torch.cat([sin_thetas, cos_thetas, zeros_v], dim=2),
            ],
            dim=1,
        )
        grid = F.affine_grid(As, img_pad.size(), align_corners=False)
        img_pad_affine = F.grid_sample(img_pad, grid, align_corners=False)

        As = torch.cat(
            [
                torch.cat([ones_v, zeros_v, tx], dim=2),
                torch.cat([zeros_v, ones_v, ty], dim=2),
            ],
            dim=1,
        )
        grid = F.affine_grid(As, img_pad_affine.size(), align_corners=False)
        img_pad_affine = F.grid_sample(img_pad_affine,
                                       grid,
                                       align_corners=False)

        # ii = (1-torch.squeeze(img_pad_affine[10,0,:,:]).detach().cpu().numpy())*255

        return img_pad_affine
