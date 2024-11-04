'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:01:48
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 11:27:51

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.utils import AffinePatch
from other_models.DeepPrint.DeepPrint import DeepPrint_stn
from other_models.PFVNet.PFVNet_AlignNet import AlignNet
from other_models.PFVNet.utils import (AffinePatch, TransformAffinePred,
                                       load_model)
from other_models.RidgeNet.RidgeNet import RidgeNet

ndim_feat = 192
align = True


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


def load_model_dp(model, ckp_path, by_name=False):

    def remove_module_string(k):
        items = k.split(".")
        idx = items.index("module")
        items = items[0:idx] + items[idx + 1:]
        return ".".join(items)

    if isinstance(ckp_path, str):
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        try:
            ckp_model_dict = ckp["model"]
        except:
            ckp_model_dict = ckp
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {
            remove_module_string(k): v
            for k, v in ckp_model_dict.items()
        }
    if by_name:
        model_dict = model.state_dict()
        state_dict = {
            k: v
            for k, v in ckp_model_dict.items() if k in model_dict
        }
        model_dict.update(state_dict)
        ckp_model_dict = model_dict

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)


if __name__ == "__main__":
    set_seed(7)
    cuda_ids = [0]
    batch_size = 1

    enh_model_path = "./ckpts/RidgeNet/best.pth"
    align_pth_path = "./ckpts/PFVNet/alignnet.pth"
    pth_path = "./ckpts/DeepPrint/best.pth.tar"

    patch_size = 160
    pad_width = 100  # for visualization

    data_dir = "./examples/data/"
    ftitle_lst = ["0", "1", "2", "3"]
    save_dir = "./examples/results/DeepPrint/"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:{}".format(str(cuda_ids[0])) if torch.cuda.
                          is_available() else "cpu")

    align_model = AlignNet()
    model = DeepPrint_stn()
    enh_model = RidgeNet()

    align_model.load_state_dict(torch.load(align_pth_path))
    load_model_dp(model, pth_path)
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

            # --- align & pad
            align_pred = align_model([enh1, enh2])
            enh2 = affine_func(enh2, transform_affine_func(align_pred))
            pad_h = max(0, 285 - input1.shape[2])
            pad_w = max(0, 285 - input1.shape[3])
            input1 = F.pad(input1, (0, pad_w, 0, pad_h))
            input2 = F.pad(input2, (0, pad_w, 0, pad_h))

            # --- input into descriptor network
            output1 = model.module.get_embedding(input1, align)
            if isinstance(output1, tuple):
                feat1 = output1[0].detach().cpu().numpy()
            else:
                feat1 = output1.detach().cpu().numpy()  # fixed the bug

            output2 = model.module.get_embedding(input2, align)
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
