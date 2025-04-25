"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 15:18:11
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 15:18:14

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import copy
import os
import os.path as osp
import random
import sys
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from skimage import morphology


def perlin(shape, frequency=5, seed=0):
    width = np.linspace(0, frequency * 3, shape[0], endpoint=False)
    height = np.linspace(0, frequency * 3, shape[1], endpoint=False)
    x, y = np.meshgrid(width, height)
    # permutation table
    # np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # integer part
    x_int = x.astype(int) % 256
    y_int = y.astype(int) % 256
    # fraction part
    x_frac = x - x_int
    y_frac = y - y_int
    # ease transitions with sigmoid-type function
    fade_x = fade(x_frac)
    fade_y = fade(y_frac)
    # noise components
    n00 = gradient(p[p[x_int] + y_int], x_frac, y_frac)
    n01 = gradient(p[p[x_int] + y_int + 1], x_frac, y_frac - 1)
    n11 = gradient(p[p[x_int + 1] + y_int + 1], x_frac - 1, y_frac - 1)
    n10 = gradient(p[p[x_int + 1] + y_int], x_frac - 1, y_frac)
    # combine noises
    x1 = lerp(n00, n10, fade_x)
    x2 = lerp(n01, n11, fade_x)
    return lerp(x1, x2, fade_y)


def intensity_normalization(img, mask=None, norm_type="min-max"):
    """map intensity to [0,1]

    Parameters:
        [None]
    Returns:
        [None]
    """
    if norm_type == "min-max":
        if mask is not None:
            img = (img * 1.0 - img[mask > 0].min()) / (
                img[mask > 0].max() - img[mask > 0].min()
            ).clip(1e-6, None)
        else:
            img = (img * 1.0 - img.min()) / (img.max() - img.min()).clip(1e-6, None)
    elif norm_type == "mean-std":
        if mask is not None:
            img = (img * 1.0 - img[mask > 0].mean()) / img[mask > 0].std().clip(
                1e-6, None
            )
        else:
            img = (img * 1.0 - img.mean()) / img.std().clip(1e-6, None)
    img = img.clip(0, 1)
    return img


def sensor_noise(
    img, mask=None, do_wet=False, stride=8, pL=0.25, tB=20, k=3, strength=128
):
    # border noise
    if mask is None:
        # mask = segmentation_coherence(img, win_size=16, stride=stride)
        mask = np.ones_like(img)
    else:
        mask = morphology.binary_dilation(mask, np.ones([3, 3]))
    dist = distance_transform_edt(mask) * stride
    dist = zoom(
        dist,
        [img.shape[0] * 1.0 / mask.shape[0], img.shape[1] * 1.0 / mask.shape[1]],
        order=1,
    )
    mask = dist > 0
    dist = np.where(dist > tB, 0, (tB - dist) / tB)
    p_border = pL * (1 + dist**3)
    # perlin noise
    p_perlin = 0
    for ii in [0, 1, 2]:
        p_perlin += 2.0 ** (-ii) * perlin(img.shape[::-1], frequency=2**ii, seed=809)
    p_perlin = intensity_normalization(p_perlin)

    # add blob noise
    img_n = img
    cur_p = np.random.random(p_perlin.shape)

    p_total = p_border * (1 + p_perlin**k)
    p_total = intensity_normalization(p_total * mask)
    # cur_p = np.random.random(p_total.shape)
    blob_noise = (cur_p <= 1.0 * p_total) * mask
    # blob_noise = morphology.binary_dilation(blob_noise)
    blob_noise = morphology.binary_closing(blob_noise)

    if do_wet is True:
        img_n = np.where(blob_noise | (1 - mask), 0, img_n)
        img_n = np.rint(np.maximum(0, img_n - strength * p_perlin)).astype(np.uint8)
    else:
        img_n = np.where(blob_noise | (1 - mask), 255, img_n)
        img_n = np.rint(np.minimum(255, img_n + strength * p_perlin)).astype(np.uint8)

    # img_n = intensity_normalization(img_n)

    return img_n, blob_noise  # intensity_normalization(blob_noise * 1)


def lerp(a, b, x):
    return a + x * (b - a)


def fade(t):
    t_squared = t**2  # Time saver
    return (6 * t_squared - 15 * t + 10) * t * t_squared


def gradient(h, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def dryness(img, selem=np.ones([2, 2])):
    img_n = morphology.dilation(img, selem)
    return img_n


def heavypress(img, selem=np.ones([3, 3])):
    img_n = morphology.erosion(img, selem)
    return img_n


def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src
    vmax, vmin = np.max(src), np.min(src)
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(1, src.shape[0] - 2)
        randY = random.randint(1, src.shape[1] - 2)
        NoiseImg[randX - 1 : randX + 1, randY - 1 : randY + 1] = NoiseImg[
            randX - 1 : randX + 1, randY - 1 : randY + 1
        ] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < vmin:
            NoiseImg[randX, randY] = vmin
        elif NoiseImg[randX, randY] > vmax:
            NoiseImg[randX, randY] = vmax
    return NoiseImg
