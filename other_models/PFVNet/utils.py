'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 10:05:38
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 10:27:12

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformAffinePred(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self, pred):
        pred_b1 = torch.Tensor(pred[:, 0])
        pred_b2 = torch.Tensor(pred[:, 1])
        cosT = torch.div(
            pred_b1, torch.sqrt(torch.square(pred_b1) + torch.square(pred_b2)))
        sinT = torch.div(
            pred_b2, torch.sqrt(torch.square(pred_b1) + torch.square(pred_b2)))

        pred[:, 0] = cosT
        pred[:, 1] = sinT
        return pred


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


def load_model(model, ckp_path):

    def remove_module_string(k):
        items = k.split(".")
        items = items[0:1] + items[2:]
        return ".".join(items)

    if isinstance(ckp_path, str):
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        ckp_model_dict = ckp["model"]
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {
            remove_module_string(k): v
            for k, v in ckp_model_dict.items()
        }

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)
