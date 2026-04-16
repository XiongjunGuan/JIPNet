'''
Description: 
Author: Xiongjun Guan
Date: 2023-12-01 14:35:11
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2026-04-16 11:31:26

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import os
from glob import glob
from multiprocessing import Pool as ThreadPool
from os import path as osp

import cv2
import numpy as np
from skimage import morphology

from fptools.fp_segmtation import segmentation_coherence


def list_generator(ftitle_lst, ext="png"):
    """ generator of list

    Parameters:
        [None]
    Returns:
        (p_name, f_name)
    """

    for i in range(len(ftitle_lst)):
        ftitle = ftitle_lst[i]

        yield ftitle, 1


def init_func(_img_dir, _res_dir):
    global img_dir
    global res_dir

    img_dir = _img_dir
    res_dir = _res_dir


def feature_extraction(img_name, tmp,pad=32):
    global img_dir
    global res_dir

    img = cv2.imread(osp.join(img_dir, img_name + '.png'), 0)

    img_pad = cv2.copyMakeBorder(
    img,
    pad, pad, pad, pad,
    borderType=cv2.BORDER_CONSTANT,
    value=255
)
    
    mask_pad = segmentation_coherence(img_pad, win_size=16, stride=8, convex=True)
    

    kernel_size = 90
    selem = np.ones((kernel_size, kernel_size))
    mask_pad = morphology.binary_erosion(mask_pad.astype(bool), selem)
    mask = mask_pad[pad:-pad, pad:-pad]
    cv2.imwrite(osp.join(res_dir, img_name + ".png"), np.uint8(mask  * 255))

    return


if __name__ == "__main__":
    img_dir = "./data_affine/bimg/"
    res_dir = "./data_affine/mask_erode/"

    if (not osp.exists(res_dir)):
        os.makedirs(res_dir)

    img_lst = glob(osp.join(img_dir, "*.png"))
    img_lst = [osp.basename(x).replace(".png", "") for x in img_lst]

    img_format = "png"
    mnt_format = "ISO"
    with ThreadPool(processes=16,
                    initializer=init_func,
                    initargs=(img_dir, res_dir)) as tp:
        tp.starmap(feature_extraction, list_generator(img_lst, ext=img_format))
