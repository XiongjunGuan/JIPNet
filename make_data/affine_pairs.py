'''
Description: 
Author: Guan Xiongjun
Date: 2022-10-09 17:51:01
LastEditTime: 2026-04-16 11:29:54
'''
import argparse
import os
import os.path as osp
import shutil
import sys
import time
from glob import glob
from multiprocessing import Pool as ThreadPool

import cv2
import numpy as np
import scipy.io as scio
from tqdm import tqdm

from fptools.fp_verifinger import Verifinger, load_minutiae


def affine_pairs(img_dir,
                 img1_title,
                 img2_title,
                 feat_dir,
                 feat1_title,
                 feat2_title,
                 save_img_dir,
                 ext="bmp"):
    global tool
    try:
        if not osp.exists(osp.join(feat_dir, feat1_title + '.bin')):
            raise ValueError("{}-{}: No Feature 1".format(
                img1_title, img2_title))
        if not osp.exists(osp.join(feat_dir, feat2_title + '.bin')):
            raise ValueError("{}-{}: No Feature 2".format(
                img1_title, img2_title))
        score, init_minu_pairs = tool.fingerprint_matching_single(
            feat_dir, feat1_title, feat_dir, feat2_title)

        img1_path = osp.join(img_dir, img1_title + f'.{ext}')
        img2_path = osp.join(img_dir, img2_title + f'.{ext}')

        MINU1 = load_minutiae(osp.join(feat_dir, feat1_title + ".mnt"))
        MINU2 = load_minutiae(osp.join(feat_dir, feat2_title + ".mnt"))

        pad_value = 300
        MINU1[:, 0] += pad_value
        MINU1[:, 1] += pad_value
        MINU2[:, 0] += pad_value
        MINU2[:, 1] += pad_value

        img1 = cv2.imread(img1_path, 0)
        img2 = cv2.imread(img2_path, 0)

        img1 = np.pad(img1, ((pad_value, pad_value), (pad_value, pad_value)),
                      mode='constant',
                      constant_values=255)
        img2 = np.pad(img2, ((pad_value, pad_value), (pad_value, pad_value)),
                      mode='constant',
                      constant_values=255)

        if init_minu_pairs.shape[0] < 5:
            raise ValueError("{}-{}: Too few match minutiae pairs".format(
                img1_title, img2_title))

        H, mask = cv2.estimateAffinePartial2D(MINU2[init_minu_pairs[:, 1],
                                                    0:2],
                                              MINU1[init_minu_pairs[:, 0],
                                                    0:2],
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=10.0)
        sx = np.sign(H[0, 0]) * np.sqrt(H[0, 0]**2 + H[0, 1]**2)
        sy = np.sign(H[1, 1]) * np.sqrt(H[1, 0]**2 + H[1, 1]**2)
        s = (sx + sy) / 2
        if abs(1 - s) > 0.15:
            raise ValueError(
                '{}-{}: Scaling size changes too much during rigid alignment !'
                .format(img1_title, img2_title))
        mask = mask.reshape((-1, ))
        init_minu_pairs = init_minu_pairs.take(np.where(mask == 1)[0], 0)

        img2_affine = cv2.warpAffine(img2,
                                     H, (img2.shape[1], img2.shape[0]),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)


        cv2.imwrite(osp.join(save_img_dir, img2_title + '.png'), img2_affine)

    except Exception as e:
        print(e)

    return


def init_func():
    global tool
    tool = Verifinger()


def list_generator(img_dir, feat_dir, save_img_dir, ext="bmp"):
    """ generator of list

    Parameters:
        [None]
    Returns:
        (p_name, f_name)
    """
    img_name_lst = glob(osp.join(img_dir, f"*_1.{ext}"))
    finger_name_lst = [
        osp.basename(x).replace(f"_1.{ext}", "") for x in img_name_lst
    ]
    # img_name_lst.sort()

    for finger_name in finger_name_lst:
        for i in range(1, 9):
            img1_title = finger_name + '_1'
            img2_title = finger_name + '_{}'.format(str(i))
            feat1_title = 'mf' + finger_name + '_1'
            feat2_title = 'mf' + finger_name + '_{}'.format(str(i))
            yield img_dir, img1_title, img2_title, feat_dir, feat1_title, feat2_title, save_img_dir, ext


if __name__ == "__main__":
    db_lst = [
        "/disk1/finger/FVC2000/DB1_A/",
        "/disk1/finger/FVC2000/DB2_A/",
        "/disk1/finger/FVC2000/DB3_A/",
        "/disk1/finger/FVC2002/DB1_A/",
        "/disk1/finger/FVC2002/DB2_A/",
        "/disk1/finger/FVC2002/DB3_A/",
        "/disk1/finger/FVC2004/DB2_A/",
    ]

    save_db_lst = [
        "/disk1/finger/FVC2000/DB1_A_affine/",
        "/disk1/finger/FVC2000/DB2_A_affine/",
        "/disk1/finger/FVC2000/DB3_A_affine/",
        "/disk1/finger/FVC2002/DB1_A_affine/",
        "/disk1/finger/FVC2002/DB2_A_affine/",
        "/disk1/finger/FVC2002/DB3_A_affine/",
        "/disk1/finger/FVC2004/DB2_A_affine/",
    ]

    for db_name, save_db_name in zip(db_lst, save_db_lst):
        img_dir = osp.join(db_name, 'img/')
        feat_dir = osp.join(db_name, 'mnt/')
        save_img_dir = osp.join(save_db_name, 'img/')

        if not osp.exists(save_img_dir):
            os.makedirs(save_img_dir)

        with ThreadPool(processes=8, initializer=init_func, initargs=()) as tp:
            tp.starmap(affine_pairs,
                       list_generator(img_dir, feat_dir, save_img_dir, "tif"))

   
