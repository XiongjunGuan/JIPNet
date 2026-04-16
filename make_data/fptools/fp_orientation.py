"""
This file (fp_orientation.py) is designed for:
    functions for fingerprint orientation field
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import numpy as np
import scipy.ndimage as ndi


def calc_orientation_graident(img, win_size=16, stride=8):
    # img = exposure.equalize_adapthist(img) * 255
    Gx, Gy = np.gradient(img.astype(np.float32))
    Gxx = ndi.gaussian_filter(Gx ** 2, win_size / 4)
    Gyy = ndi.gaussian_filter(Gy ** 2, win_size / 4)
    Gxy = ndi.gaussian_filter(-Gx * Gy, win_size / 4)
    coh = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2)  # / (Gxx + Gyy).clip(1e-6, None)
    if stride != 1:
        Gxx = ndi.uniform_filter(Gxx, stride)[::stride, ::stride]
        Gyy = ndi.uniform_filter(Gyy, stride)[::stride, ::stride]
        Gxy = ndi.uniform_filter(Gxy, stride)[::stride, ::stride]
        coh = ndi.uniform_filter(coh, stride)[::stride, ::stride]
    ori = np.arctan2(2 * Gxy, Gxx - Gyy) * 90 / np.pi
    return ori, coh


def minus_orientation(ori, anchor):
    ori = ori - anchor
    ori = np.where(ori >= 90, ori - 180, ori)
    ori = np.where(ori < -90, ori + 180, ori)
    return ori


def zoom_orientation(ori, scale):
    cos_2ori = np.cos(ori * np.pi / 90)
    sin_2ori = np.sin(ori * np.pi / 90)
    cos_2ori = ndi.zoom(cos_2ori, scale, order=1)
    sin_2ori = ndi.zoom(sin_2ori, scale, order=1)
    ori = np.arctan2(sin_2ori, cos_2ori) * 90 / np.pi
    return ori


def transform_to_reference(arr, pose, tar_shape=None, order=0, cval=0, factor=8, is_ori=False):
    """transform array to standard pose

    Parameters:
        [None]
    Returns:
        [None]
    """
    x, y, theta = pose
    tar_shape = np.array(arr.shape[-2:]) if tar_shape is None else tar_shape
    center = tar_shape // 2
    finger_center = np.array([y, x]).astype(np.float32) / factor
    sin_theta = np.sin(theta * np.pi / 180.0)
    cos_theta = np.cos(theta * np.pi / 180.0)
    mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    indices = np.stack(np.meshgrid(*[np.arange(x) for x in tar_shape], indexing="ij")).astype(np.float32)
    coord_indices = indices[-2:].reshape(2, -1)
    coord_indices = np.dot(mat, coord_indices - center[:, None]) + finger_center[:, None]
    indices[-2:] = coord_indices.reshape(2, *tar_shape)
    if is_ori:
        new_arr = arr + theta
        cos_2angle = ndi.map_coordinates(np.cos(2 * new_arr * np.pi / 180), indices, order=order, mode="nearest")
        sin_2angle = ndi.map_coordinates(np.sin(2 * new_arr * np.pi / 180), indices, order=order, mode="nearest")
        new_arr = np.arctan2(sin_2angle, cos_2angle) * 180 / np.pi / 2
    else:
        new_arr = ndi.map_coordinates(arr, indices, order=order, cval=cval)
    return new_arr


def transform_to_target(arr, pose, tar_shape=None, factor=8.0, angle=False, order=1):
    """transform array to target pose

    Parameters:
        [None]
    Returns:
        [None]
    """
    x, y, theta = pose
    tar_shape = np.array(arr.shape[-2:]) if tar_shape is None else tar_shape
    center = np.array(arr.shape[-2:]) // 2
    finger_center = np.array([y, x]).astype(np.float32) / factor
    sin_theta = np.sin(theta * np.pi / 180.0)
    cos_theta = np.cos(theta * np.pi / 180.0)
    mat = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    indices = np.stack(np.meshgrid(*[np.arange(x) for x in tar_shape], indexing="ij")).astype(np.float32)
    coord_indices = indices[-2:].reshape(2, -1)
    coord_indices = np.dot(mat, coord_indices - finger_center[:, None]) + center[:, None]
    indices[-2:] = coord_indices.reshape(2, *tar_shape)
    if angle:
        new_arr = arr - theta
        cos_2angle = ndi.map_coordinates(np.cos(2 * new_arr * np.pi / 180), indices, order=1, mode="nearest")
        sin_2angle = ndi.map_coordinates(np.sin(2 * new_arr * np.pi / 180), indices, order=1, mode="nearest")
        new_arr = np.arctan2(sin_2angle, cos_2angle) * 180 / np.pi / 2
    else:
        new_arr = ndi.map_coordinates(arr, indices, order=order, mode="nearest")
    return new_arr
