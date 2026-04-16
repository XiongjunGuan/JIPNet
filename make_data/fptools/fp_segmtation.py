"""
This file (fp_segmtation.py) is designed for:
    functions for fingerprint segmentation
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
from skimage import morphology
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from skimage.segmentation import watershed
from skimage.filters import sobel
import torch
import torch.nn.functional as F
import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
from fp_orientation import calc_orientation_graident


def ridge_coherence(img, win_size=8, stride=8):
    Gx, Gy = np.gradient(img.astype(np.float32))
    coh = np.sqrt((Gx ** 2 - Gy ** 2) ** 2 + 4 * Gx ** 2 * Gy ** 2)
    coh = (
        F.avg_pool2d(torch.tensor(coh)[None, None].float(), 2 * win_size - 1, stride=stride, padding=win_size - 1)
        .squeeze()
        .numpy()
    )
    return coh


def ridge_intensity(img, win_size=8, stride=8):
    coh = (
        F.avg_pool2d(torch.tensor(img)[None, None].float(), 2 * win_size - 1, stride=stride, padding=win_size - 1)
        .squeeze()
        .numpy()
    )
    return coh


def segmentation_clustering(samples):
    """segmentation using cluster based method

    Parameters:
        samples: (N, M)
    Returns:
        [None]
    """
    pred_1 = samples[:, 0] > 20
    pred_2 = samples[:, 1] < samples[pred_1, 1].mean()
    return pred_1 | pred_2


def find_largest_connected_region(seg):
    label_im, nb_labels = ndi.label(seg)
    sizes = ndi.sum_labels(seg, label_im, index=range(0, nb_labels + 1))
    max_label = np.argmax(sizes)
    largest_seg = np.zeros_like(seg)
    largest_seg[label_im == max_label] = 1
    return largest_seg


def convex_hull_image(data):
    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v, 0], region[v, 1]) for v in hull.vertices]
    img = Image.new("L", data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)
    return mask.T


def segmentation_postprocessing(seg, kernel_size=5, stride=8, convex=False):
    selem = np.ones((kernel_size, kernel_size))
    # seg = morphology.binary_opening(seg.astype(np.bool), footprint=selem)
    # seg = morphology.binary_closing(seg.astype(np.bool), footprint=selem)
    seg = morphology.binary_erosion(seg.astype(bool), selem)
    seg = morphology.remove_small_holes(seg, area_threshold=15000 // (stride ** 2))
    seg = morphology.remove_small_objects(seg, min_size=15000 // (stride ** 2))
    seg = morphology.binary_dilation(seg.astype(bool), selem)
    seg = find_largest_connected_region(seg)
    if convex and seg.sum() > 0:
        seg = convex_hull_image(seg)
    return seg


def segmentation_coherence(img, win_size=16, stride=8, threshold=50, convex=False, resize=False):
    # average pooling
    _, coh = calc_orientation_graident(img, win_size, stride)
    if stride != 1:
        coh = ndi.zoom(coh, (img.shape[0] * 1.0 / coh.shape[0], img.shape[1] * 1.0 / coh.shape[1]), order=1)
    seg = coh > threshold

    seg = segmentation_postprocessing(seg, kernel_size=15, stride=1, convex=convex)
    return seg


def segmentation_watershed(img, markers, stride=8):
    img = ndi.zoom(img, 1.0 / stride, order=1)
    # markers = zoom(markers, 1.0 / stride, order=0)
    elevation_map = sobel(img)
    seg = watershed(elevation_map, markers) - 1
    # seg = morphology.remove_small_holes(seg, area_threshold=100)
    # seg = morphology.remove_small_objects(seg, min_size=100)
    return seg


if __name__ == "__main__":
    prefix = ""
