"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 15:17:10
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 15:17:18

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
import torch
from scipy.ndimage import rotate, shift, zoom
from torch.utils.data import DataLoader, Dataset

from augmentation import img_aug
from utils import AffinePatch


def show_pairs(bimg1, bimg2):
    h, w = bimg1.shape
    img = np.ones((h, w, 3)) * 255
    b1 = bimg1 < 128
    b2 = bimg2 < 128
    common = b1 & b2
    only1 = b1 & (~b2)
    only2 = (~b1) & b2

    img[:, :, 0][common] = 0
    img[:, :, 1][~common] = 255
    img[:, :, 2][common] = 0

    img[:, :, 0][only1] = 128
    img[:, :, 1][only1] = 128
    img[:, :, 2][only1] = 128

    img[:, :, 0][only2] = 0
    img[:, :, 1][only2] = 0
    img[:, :, 2][only2] = 255

    return img


def cut_patch(arr, info, patch_size=0):
    arr = copy.deepcopy(arr)
    arr = np.pad(
        arr,
        ((patch_size, patch_size), (patch_size, patch_size)),
        "constant",
        constant_values=255,
    )
    info = np.array(info)
    info[0:2] += patch_size

    rows, cols = arr.shape[:2]
    M = cv2.getRotationMatrix2D((int(info[1]), int(info[0])), info[2], 1)
    rotated_image = cv2.warpAffine(
        arr, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    y = int(info[0] - patch_size / 2)
    x = int(info[1] - patch_size / 2)
    x_end = x + patch_size
    y_end = y + patch_size

    # Crop the rotated image
    cropped_image = rotated_image[y:y_end, x:x_end]

    return cropped_image


def restore_img(arr, restore_info, borderValue=255):
    arr = copy.deepcopy(arr)
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), restore_info[2], 1)
    M[0, 2] += restore_info[1]
    M[1, 2] += restore_info[0]

    img = cv2.warpAffine(
        arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue
    )

    return img


def reverse_info(info1, info2):
    rinfo2 = np.array(info2) - np.array(info1)
    rinfo2[2] = -rinfo2[2]
    y, x = rinfo2[0], rinfo2[1]
    theta = np.deg2rad(info1[2])
    rinfo2[0] = -x * np.sin(theta) + y * np.cos(theta)
    rinfo2[1] = x * np.cos(theta) + y * np.sin(theta)
    return rinfo2


class load_dataset_train(Dataset):

    def __init__(
        self,
        info_lst: list,
        patch_size=128,
        use_augmentation=True,
        img_augmentation=False,
        need_name=False,
    ):
        self.info_lst = info_lst
        self.patch_size = patch_size
        self.use_augmentation = use_augmentation
        self.img_augmentation = img_augmentation
        self.need_name = need_name

    def __len__(self):
        return len(self.info_lst)

    def __getitem__(self, idx):
        info_path = self.info_lst[idx]

        with open(info_path, "r") as file:
            lines = file.readlines()
            fpath1 = lines[1].replace("\n", "")
            fpath2 = lines[2].replace("\n", "")
            info1 = np.array(list(map(float, lines[3].strip().split())))
            info2 = np.array(list(map(float, lines[4].strip().split())))
            gt = list(map(float, lines[5].strip().split()))[0]

        # augmentation
        if self.use_augmentation is True:
            if random.random() > 0.5:
                fpath1, fpath2 = fpath2, fpath1
                info1, info2 = info2, info1

        patch_img1 = cv2.imread(fpath1, 0).astype(np.float32)
        patch_img2 = cv2.imread(fpath2, 0).astype(np.float32)

        if self.img_augmentation is True:
            if random.random() > 0.5:
                patch_img1 = img_aug(patch_img1)
            if random.random() > 0.5:
                patch_img2 = img_aug(patch_img2)

        h, w = patch_img1.shape
        hc, wc = h // 2, w // 2
        hps = self.patch_size // 2
        patch_img1 = patch_img1[hc - hps : hc + hps, wc - hps : wc + hps]
        patch_img2 = patch_img2[hc - hps : hc + hps, wc - hps : wc + hps]

        rinfo2 = reverse_info(info1, info2)

        align_target = np.array(
            [
                np.cos(np.deg2rad(rinfo2[2])),
                np.sin(np.deg2rad(rinfo2[2])),
                rinfo2[1],
                rinfo2[0],
            ],
            dtype=np.float32,
        )

        patch_img1 = ((255.0 - patch_img1) / 255.0)[np.newaxis, :, :]
        patch_img2 = ((255.0 - patch_img2) / 255.0)[np.newaxis, :, :]

        target = np.array([gt, 1 - gt], dtype=np.float32)

        if self.need_name is True:
            return patch_img1, patch_img2, align_target, target, fpath1, fpath2
        else:
            return patch_img1, patch_img2, align_target, target


def get_dataloader_train(
    info_lst: list,
    batch_size=1,
    patch_size=128,
    use_augmentation=True,
    img_augmentation=False,
    shuffle=True,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            info_lst,
            patch_size=patch_size,
            use_augmentation=use_augmentation,
            img_augmentation=img_augmentation,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True
    )
    logging.info(f"n_train:{len(dataset)}")

    return train_loader


def get_dataloader_valid(
    info_lst: list,
    patch_size=128,
    batch_size=1,
    use_augmentation=False,
    need_name=False,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            info_lst,
            patch_size=patch_size,
            use_augmentation=use_augmentation,
            need_name=need_name,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    valid_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    logging.info(f"n_valid:{len(dataset)}")

    return valid_loader
