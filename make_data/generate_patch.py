'''
Description: 
Author: Xiongjun Guan
Date: 2023-12-01 18:17:17
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2026-04-16 11:34:50

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import itertools
import os
import random
from glob import glob
from os import path as osp

import cv2
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

random.seed(7)


def save_info(save_img1_path, save_img2_path, info1, info2, gt, save_path):
    with open(osp.join(save_path), "w") as fp:
        fp.write(
            f"File for patch_info. from top to left: img_path1/2, info1/2, gt, info from left to right: row col theta.\n"
        )
        fp.write(f"{save_img1_path}\n")
        fp.write(f"{save_img2_path}\n")
        fp.write(" ".join(["{:.4f}".format(x) for x in info1]))
        fp.write("\n")
        fp.write(" ".join(["{:.4f}".format(x) for x in info2]))
        fp.write("\n")
        fp.write(f"{gt}\n")


def cut_patch(arr, info, patch_size=128):
    arr = np.pad(arr, ((patch_size, patch_size), (patch_size, patch_size)),
                 'constant',
                 constant_values=255)
    info = np.array(info)
    info[0:2] += patch_size

    rows, cols = arr.shape[:2]
    M = cv2.getRotationMatrix2D((int(info[1]), int(info[0])), info[2], 1)
    rotated_image = cv2.warpAffine(arr,
                                   M, (cols, rows),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)

    y = int(info[0] - patch_size / 2)
    x = int(info[1] - patch_size / 2)
    x_end = x + patch_size
    y_end = y + patch_size

    # Crop the rotated image
    cropped_image = rotated_image[y:y_end, x:x_end]

    return cropped_image

def sample_gaussian_points(mask, center, variance):
    flattened_mask = mask.flatten()

    # Get indices of non-zero pixels in the mask
    non_zero_indices = np.nonzero(flattened_mask)[0]

    # Convert indices to 2D coordinates
    coordinates = np.unravel_index(non_zero_indices, mask.shape)

    # Convert coordinates to be relative to the given center
    relative_coordinates = np.transpose(np.array(coordinates)) - center

    # Compute the probability density values using a 2D Gaussian distribution
    probabilities = np.exp(
        -np.sum((relative_coordinates / 100) ** 2, axis=1) / (2 * variance)
    )

    # Sample indices based on the computed probabilities
    sampled_indices = np.random.choice(
        non_zero_indices,
        size=1,
        p=probabilities / np.sum(probabilities)
    )

    # Convert sampled flat indices back to 2D mask coordinates
    sampled_indices = np.unravel_index(sampled_indices, mask.shape)

    return np.array(sampled_indices).squeeze()

def sample_mask_points(mask):
    flattened_mask = mask.flatten()

    # Get indices of non-zero pixels in the mask
    non_zero_indices = np.nonzero(flattened_mask)[0]

    # Assign uniform probabilities to all valid pixels
    probabilities = np.ones_like(non_zero_indices)

    # Sample indices based on uniform probability
    sampled_indices = np.random.choice(
        non_zero_indices,
        size=1,
        p=probabilities / np.sum(probabilities)
    )

    # Convert sampled flat indices back to 2D mask coordinates
    sampled_indices = np.unravel_index(sampled_indices, mask.shape)

    return np.array(sampled_indices).squeeze()

def generate_pos_single(common_mask, var1=1.0):
    h, w = common_mask.shape

    rows, cols = np.nonzero(common_mask)
    center = np.array([np.mean(rows), np.mean(cols)])
    pos1 = sample_gaussian_points(common_mask, center=center, variance=var1)

    return pos1


def generate_pos(common_mask, ring_r=50, var1=1.0, var2=2.0):
    h, w = common_mask.shape

    rows, cols = np.nonzero(common_mask)
    center = np.array([np.mean(rows), np.mean(cols)])
    pos1 = sample_gaussian_points(common_mask, center=center, variance=var1)


    h, w = common_mask.shape
    x, y = np.meshgrid(np.array(np.arange(0, w)),
                       np.meshgrid(np.arange(0, h)),
                       indexing='xy')
    dis = np.sqrt((x - pos1[1])**2 + (y - pos1[0])**2)

    ring_mask = (dis > ring_r - 15) & (dis < ring_r + 15)
    ring_mask = ring_mask * common_mask
    pos2 = sample_gaussian_points(ring_mask, center=center, variance=var2)

    pos_ring_arrs = np.where(ring_mask > 0)
    idx = random.randint(0, len(pos_ring_arrs[0]) - 1)
    pos2 = np.array([pos_ring_arrs[0][idx], pos_ring_arrs[1][idx]])

    return pos1, pos2


def make_data(ftitle_lst, iter=2, ring_r_max=80, patch_size=128):
    global img_dir
    global mask_dir
    global res_img_dir
    global res_info_dir

    generate_idx = 0

    for ftitle in tqdm(ftitle_lst):
        # genuine
        finger = ftitle.split("_")[0]

        gm_ftitle_lst = glob(osp.join(mask_dir, f"{finger}_*.png"))
        gm_ftitle_lst = [
            osp.basename(x).replace(".png", "") for x in gm_ftitle_lst
        ]

        for gm_ftitle in gm_ftitle_lst:
            if gm_ftitle == ftitle:
                continue

            mask1 = cv2.imread(osp.join(mask_dir, ftitle + ".png"), 0)
            mask2 = cv2.imread(osp.join(mask_dir, gm_ftitle + ".png"), 0)

            if (mask1 is None) or (mask2 is None):
                continue

            mask1 = mask1 > 0
            mask2 = mask2 > 0
            common_mask = mask1 * mask2
            if (np.sum(common_mask) < 100):
                continue

            for i in range(iter):
                # genuine
                ring_r = random.randint(0, ring_r_max)
                try:
                    pos1, pos2 = generate_pos(common_mask, ring_r=ring_r)
                except:
                    continue
                ang1 = random.randint(-180, 180)
                ang2 = random.randint(-180, 180)

                info1 = np.array([pos1[0], pos1[1], ang1])
                info2 = np.array([pos2[0], pos2[1], ang2])

                img1 = cv2.imread(osp.join(img_dir, ftitle + ".png"),
                                  0).astype(np.float32)
                img2 = cv2.imread(osp.join(img_dir, gm_ftitle + ".png"),
                                  0).astype(np.float32)

                patch_img1 = cut_patch(img1, info1, patch_size)
                patch_img2 = cut_patch(img2, info2, patch_size)

                save_path = osp.join(res_info_dir, f"{generate_idx}.txt")
                save_img1_path = osp.join(res_img_dir, f"{generate_idx}_1.png")
                save_img2_path = osp.join(res_img_dir, f"{generate_idx}_2.png")

                cv2.imwrite(save_img1_path, patch_img1)
                cv2.imwrite(save_img2_path, patch_img2)

                save_info(save_img1_path, save_img2_path, info1, info2, 1,
                          save_path)

                generate_idx += 1

                # impostor
                while True:
                    idx = random.randint(0, len(ftitle_lst) - 1)
                    im_ftitle = ftitle_lst[idx]
                    im_finger = im_ftitle.split("_")[0]
                    if not im_finger == finger:
                        img2 = cv2.imread(
                            osp.join(img_dir, im_ftitle + ".png"),
                            0).astype(np.float32)

                        mask2_im = cv2.imread(
                            osp.join(mask_dir, im_ftitle + ".png"), 0)

                        pos2 = generate_pos_single(mask2_im)
                        ang2 = random.randint(-180, 180)
                        info2 = np.array([pos2[0], pos2[1], ang2])

                        patch_img2 = cut_patch(img2, info2, patch_size)

                        save_path = osp.join(res_info_dir,
                                             f"{generate_idx}.txt")
                        save_img1_path = osp.join(res_img_dir,
                                                  f"{generate_idx}_1.png")
                        save_img2_path = osp.join(res_img_dir,
                                                  f"{generate_idx}_2.png")

                        cv2.imwrite(save_img1_path, patch_img1)
                        cv2.imwrite(save_img2_path, patch_img2)

                        save_info(save_img1_path, save_img2_path, info1, info2,
                                  0, save_path)

                        generate_idx += 1

                        break

    return


if __name__ == "__main__":
    base_dir = "./data_affine/"
    img_dir = osp.join(base_dir, "img")
    mask_dir = osp.join(base_dir, "mask_erode")

    save_base_dir = "./data_affine/results/"
    res_img_dir = osp.join(save_base_dir, "img")
    res_info_dir = osp.join(save_base_dir, "info")

    if not osp.exists(res_img_dir):
        os.makedirs(res_img_dir)
    if not osp.exists(res_info_dir):
        os.makedirs(res_info_dir)

    ftitle_lst = glob(osp.join(mask_dir, "*.png"))
    ftitle_lst = [osp.basename(x).replace(".png", "") for x in ftitle_lst]
    ftitle_lst = sorted(
        ftitle_lst,
        key=lambda x: 100 * int(x.split("_")[0]) + int(x.split("_")[1]))

    make_data(ftitle_lst, iter=4, ring_r_max=100, patch_size=160)
