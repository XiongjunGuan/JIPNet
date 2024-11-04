'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:01:48
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 12:00:19

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.utils import AffinePatch
from other_models.AKAZE.match import matchDes


def localEqualHist(img):
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(32, 32))
    dst = clahe.apply(img)
    return dst


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


if __name__ == "__main__":
    set_seed(7)
    batch_size = 1

    patch_size = 160
    pad_width = 100  # for visualization

    data_dir = "./examples/data/"
    ftitle_lst = ["0", "1", "2", "3"]
    save_dir = "./examples/results/AKAZE/"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    affine_func = AffinePatch()
    akaze = cv2.AKAZE_create()

    with torch.no_grad():
        for ftitle in tqdm(ftitle_lst):
            img1 = cv2.imread(osp.join(data_dir, f"{ftitle}_1.png"), 0)
            img2 = cv2.imread(osp.join(data_dir, f"{ftitle}_2.png"), 0)

            patch_size1 = img1.shape[0]
            center = np.array([[patch_size // 2, patch_size // 2]])

            img1 = localEqualHist(img1)
            img2 = localEqualHist(img2)

            kp1, des1 = akaze.detectAndCompute(img1, None)
            kp2, des2 = akaze.detectAndCompute(img2, None)

            y, align_pred = matchDes(kp1, kp2, des1, des2, patch_size, center)

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
