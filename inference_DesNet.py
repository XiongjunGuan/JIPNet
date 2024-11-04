'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:01:48
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 11:28:04

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.utils import AffinePatch
from other_models.DesNet.PatchMatching import DesNet
from other_models.PFVNet.PFVNet_AlignNet import AlignNet
from other_models.PFVNet.utils import (AffinePatch, TransformAffinePred,
                                       load_model)
from other_models.RidgeNet.RidgeNet import RidgeNet


def norm_pred(align_pred, eps=1e-6):
    pred_b1 = align_pred[:, 0]
    pred_b2 = align_pred[:, 1]
    cosT = np.divide(pred_b1,
                     eps + np.sqrt(np.square(pred_b1) + np.square(pred_b2)))
    sinT = np.divide(pred_b2,
                     eps + np.sqrt(np.square(pred_b1) + np.square(pred_b2)))
    pred_norm = np.concatenate(
        [
            cosT[:, None], sinT[:, None], align_pred[:, 2][:, None],
            align_pred[:, 3][:, None]
        ],
        axis=1,
    )
    return pred_norm


def set_seed(seed=7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_desnet(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == "__main__":
    set_seed(7)
    cuda_ids = [0]
    batch_size = 1

    enh_model_path = "./ckpts/RidgeNet/best.pth"
    align_pth_path = "./ckpts/PFVNet/alignnet.pth"
    pth_path = "./ckpts/DesNet/best.pth"

    patch_size = 160
    pad_width = 100  # for visualization

    data_dir = "./examples/data/"
    ftitle_lst = ["0", "1", "2", "3"]
    save_dir = "./examples/results/DesNet/"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:{}".format(str(cuda_ids[0])) if torch.cuda.
                          is_available() else "cpu")

    align_model = AlignNet()
    model = DesNet()
    enh_model = RidgeNet()

    align_model.load_state_dict(torch.load(align_pth_path))
    load_desnet(model, pth_path)
    load_model(enh_model, enh_model_path)

    align_model = torch.nn.DataParallel(
        align_model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)
    model = torch.nn.DataParallel(
        model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)
    enh_model = torch.nn.DataParallel(
        enh_model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)

    affine_func = AffinePatch()
    transform_affine_func = TransformAffinePred()

    align_model.eval()
    model.eval()
    enh_model.eval()
    with torch.no_grad():
        for ftitle in tqdm(ftitle_lst):
            img1 = cv2.imread(osp.join(data_dir, f"{ftitle}_1.png"), 0)
            img2 = cv2.imread(osp.join(data_dir, f"{ftitle}_2.png"), 0)

            input1 = (img1 / 255.0)[np.newaxis, np.newaxis, :, :]
            input2 = (img2 / 255.0)[np.newaxis, np.newaxis, :, :]

            input1 = torch.tensor(input1).float().to(device)
            input2 = torch.tensor(input2).float().to(device)

            # --- enh
            [enh1, enh2] = enh_model([input1, input2])

            # --- align
            align_pred = align_model([enh1, enh2])

            # --- regist & pad
            input1 = 1 - input1
            input2 = 1 - input2
            input2 = affine_func(input2, transform_affine_func(align_pred))
            pad_h = max(0, 200 - input1.shape[2])
            pad_w = max(0, 200 - input1.shape[3])
            input1 = F.pad(input1, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                                    pad_h - pad_h // 2))
            input2 = F.pad(input2, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                                    pad_h - pad_h // 2))
            input1 = 1 - input1
            input2 = 1 - input2

            # --- input into descriptor network
            input1 = (input1 - 0.5) / 0.5
            input2 = (input2 - 0.5) / 0.5
            output1 = model.module.get_embedding(input1)
            output2 = model.module.get_embedding(input2)

            # --- calculate score
            if isinstance(output1, tuple):
                feat1 = output1[0].detach().cpu().numpy()
            else:
                feat1 = output1.detach().cpu().numpy()  # fixed the bug
            if isinstance(output2, tuple):
                feat2 = output2[0].detach().cpu().numpy()
            else:
                feat2 = output2.detach().cpu().numpy()  # fixed the bug

            score = (feat1 * feat2).sum(1) / (
                np.linalg.norm(feat1, axis=1) *
                np.linalg.norm(feat2, axis=1)).clip(1e-6, None)

            y = score.squeeze()
            align_pred = align_pred.squeeze().cpu().numpy()

            align_pred = norm_pred(align_pred[None, :])

            # --- visualization
            img2 = torch.tensor(np.float32(img2)[None, None, :, :] / 255.0)
            img2 = affine_func(img2,
                               torch.tensor(align_pred),
                               pad_width=pad_width).squeeze().numpy()
            img2 = np.clip(img2 * 255, 0, 255)
            img1 = np.pad(img1,
                          [[pad_width, pad_width], [pad_width, pad_width]],
                          mode='constant',
                          constant_values=0)
            img = (img1 * 1.0 + img2 * 1.0) * 0.5
            img = np.clip(img, 0, 255)
            cv2.imwrite(osp.join(save_dir, f"{ftitle}.png"), img)

            # --- save info
            save_path = osp.join(save_dir, f"{ftitle}.txt")
            with open(save_path, 'w') as file:
                file.write("1-2 matching probability:\n")

                file.write("{:.2f}\n\n".format(y))
                file.write("1-2 pose alignment (cos, sin, tx, ty):\n")
                for number in align_pred[0, :]:
                    file.write("{:.2f}, ".format(number))
                file.write("\n")
