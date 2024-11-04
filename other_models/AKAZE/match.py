'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 11:51:46
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 11:54:58

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import cv2
import numpy as np


def info2vec(info):
    vec = np.array([
        np.cos(np.deg2rad(info[2])),
        np.sin(np.deg2rad(info[2])), info[1], info[0]
    ],
                   dtype=np.float32)
    return vec


def matchDes(kp1, kp2, des1, des2, patch_size, center):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    top_matches = matches

    # ---- select top m=15 keyporints
    m = 15
    top_matches = sorted(top_matches, key=lambda x: x.distance)
    nb_img1 = np.zeros((patch_size + 40, patch_size + 40))
    nb_img2 = np.zeros((patch_size + 40, patch_size + 40))
    nb_size = 4

    closest_matches = []
    for match in top_matches:
        pt1 = np.int32(kp1[match.queryIdx].pt) + 20
        pt2 = np.int32(kp2[match.trainIdx].pt) + 20
        if (nb_img1[pt1[0], pt1[1]] == 0) and (nb_img2[pt2[0], pt2[1]] == 0):
            nb_img1[pt1[0] - nb_size:pt1[0] + nb_size,
                    pt1[1] - nb_size:pt1[1] + nb_size] = 1
            nb_img2[pt2[0] - nb_size:pt2[0] + nb_size,
                    pt2[1] - nb_size:pt2[1] + nb_size] = 1

            closest_matches.append(match)

    top_matches = closest_matches[:m]

    # ---- select top K=6 keyporints
    K = 6
    cnt = m
    minu1 = np.array([kp1[match.queryIdx].pt for match in top_matches])
    minu2 = np.array([kp2[match.trainIdx].pt for match in top_matches])
    D = np.zeros((m, m))
    E = np.zeros((m, ))
    M = np.ones((m, ))
    for i in range(0, m - 1):
        for j in range(i, m):
            D[i, j] = np.abs(
                np.linalg.norm(minu1[i] - minu1[j]) -
                np.linalg.norm(minu2[i] - minu2[j]))
            D[j, i] = D[i, j]
            E[i] += D[i, j]
            E[j] += D[i, j]
    while cnt > K:
        max_value = np.max(E[M == 1])
        max_index = np.argmax(E == max_value)
        M[max_index] = 0
        for i in range(0, m):
            if M[i] != 0:
                E[i] -= D[max_index, i]
        cnt -= 1
    top_matches = [item for item, flag in zip(top_matches, M) if flag == 1]

    # calculate score
    score_texture = np.mean([match.distance for match in top_matches])

    minu1 = np.array([kp1[match.queryIdx].pt
                      for match in top_matches]) - center
    minu2 = np.array([kp2[match.trainIdx].pt
                      for match in top_matches]) - center

    minu1 = minu1[:, [1, 0]]
    minu2 = minu2[:, [1, 0]]

    cnt = 0
    score_topo = 0
    for i in range(len(top_matches)):
        for j in range(i + 1, len(top_matches)):
            score_topo += np.abs(
                np.linalg.norm(minu1[i] - minu1[j]) -
                np.linalg.norm(minu2[i] - minu2[j]))
            cnt += 1
    score_topo /= cnt
    score = -(score_texture * 0.04 + score_topo * 0.96)

    H, mask = cv2.estimateAffinePartial2D(minu1,
                                          minu2,
                                          method=cv2.RANSAC,
                                          ransacReprojThreshold=20.0)
    mask = mask.reshape((-1, ))

    minu1 = minu1[mask == 1]
    minu2 = minu2[mask == 1]

    ang = -np.rad2deg(np.arctan2(H[1, 0], H[1, 1]))

    radian = -np.arctan2(H[1, 0], H[1, 1])
    M = np.array([[np.cos(radian), -np.sin(radian)],
                  [np.sin(radian), np.cos(radian)]])

    transformed_minu2 = minu2 @ M.T
    tx = np.mean((minu1 - transformed_minu2)[:, 0])
    ty = np.mean((minu1 - transformed_minu2)[:, 1])

    align_pred = info2vec([tx, ty, ang])

    return score, align_pred
