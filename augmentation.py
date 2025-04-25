"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 15:17:56
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 15:18:03

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import copy
import logging
import os.path as osp
import random
from glob import glob
from random import randint

import cv2
import numpy as np
import scipy.io as scio
from scipy.ndimage import rotate, shift, zoom
from torch.utils.data import DataLoader, Dataset

from add_noise import GaussianNoise, dryness, heavypress, sensor_noise


def img_aug(img):
    # ---- sensor noise: perlin
    strength = np.random.uniform(60, 120)
    if np.random.rand() < 0.5:
        img, _ = sensor_noise(
            img, stride=8, do_wet=False, pL=0.1, tB=20, strength=strength
        )
    else:
        img, _ = sensor_noise(
            img, stride=8, do_wet=True, pL=0.1, tB=20, strength=strength
        )
    # ---- sensor noise: gaussian
    if np.random.rand() < 0.5:
        noise_mode = np.random.choice([1, 2, 3], 1, p=[0.33, 0.33, 0.34])[0]
        if noise_mode == 1:
            img = dryness(img)
        elif noise_mode == 2:
            img = heavypress(img)
        elif noise_mode == 3:
            blur_core = np.random.choice([3, 5], 1, p=[0.5, 0.5])[0]
            img = cv2.GaussianBlur(img, (blur_core, blur_core), 3)
            gaussian_sigma = np.random.choice([5, 10], 1, p=[0.5, 0.5])[0]
            img = GaussianNoise(img, 0, gaussian_sigma, 0.1)

    # ---- inverse color
    if np.random.rand() < 0.5:
        img = 255 - img

    # ---- clip
    img = np.clip(img, 0, 255)

    return img
