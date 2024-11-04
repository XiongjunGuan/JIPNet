"""
Description: 
Author: Xiongjun Guan
Date: 2023-04-13 17:12:44
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-04-19 21:01:00

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata

from other_models.RidgeNet.units import *


class RidgeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.img_norm = NormalizeModule(m0=0, var0=1)

        # feature extraction VGG
        self.conv1 = nn.Sequential(ConvBnPRelu(1, 64, 3),
                                   ConvBnPRelu(64, 64, 3), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(ConvBnPRelu(64, 128, 3),
                                   ConvBnPRelu(128, 128, 3),
                                   nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            ConvBnPRelu(128, 256, 3),
            ConvBnPRelu(256, 256, 3),
            ConvBnPRelu(256, 256, 3),
            nn.MaxPool2d(2, 2),
        )

        # multi-scale ASPP
        self.conv4_1 = ConvBnPRelu(256, 256, 3, padding=1, dilation=1)
        self.ori1 = nn.Sequential(
            ConvBnPRelu(256, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 90, 1, stride=1, padding=0),
        )

        self.conv4_2 = ConvBnPRelu(256, 256, 3, padding=4, dilation=4)
        self.ori2 = nn.Sequential(
            ConvBnPRelu(256, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 90, 1, stride=1, padding=0),
        )

        self.conv4_3 = ConvBnPRelu(256, 256, 3, padding=8, dilation=8)
        self.ori3 = nn.Sequential(
            ConvBnPRelu(256, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 90, 1, stride=1, padding=0),
        )

        # enhance part
        gabor_cos, gabor_sin = gabor_bank(enh_ksize=25, ori_stride=2, Lambda=8)

        self.enh_img_real = nn.Conv2d(gabor_cos.size(1),
                                      gabor_cos.size(0),
                                      kernel_size=(25, 25),
                                      padding=12)
        self.enh_img_real.weight = nn.Parameter(gabor_cos, requires_grad=True)
        self.enh_img_real.bias = nn.Parameter(torch.zeros(gabor_cos.size(0)),
                                              requires_grad=True)

        self.enh_img_imag = nn.Conv2d(gabor_sin.size(1),
                                      gabor_sin.size(0),
                                      kernel_size=(25, 25),
                                      padding=12)
        self.enh_img_imag.weight = nn.Parameter(gabor_sin, requires_grad=True)
        self.enh_img_imag.bias = nn.Parameter(torch.zeros(gabor_sin.size(0)),
                                              requires_grad=True)

    def forward(self, input):
        is_list = isinstance(input, tuple) or isinstance(input, list)
        if is_list:
            batch_dim = input[0].shape[0]
            input = torch.cat(input, dim=0)

        img_norm = self.img_norm(input)

        # feature extraction VGG
        conv1 = self.conv1(img_norm)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # multi-scale ASPP
        conv4_1 = self.conv4_1(conv3)
        ori1 = self.ori1(conv4_1)

        conv4_2 = self.conv4_2(conv3)
        ori2 = self.ori2(conv4_2)

        conv4_3 = self.conv4_3(conv3)
        ori3 = self.ori3(conv4_3)

        ori_out = torch.sigmoid(ori1 + ori2 + ori3)

        # enhance part
        enh_real = self.enh_img_real(input)
        # enh_imag = self.enh_img_imag(input)
        ori_peak = orientation_highest_peak(ori_out)
        ori_peak = select_max_orientation(ori_peak)

        ori_up = F.interpolate(ori_peak,
                               size=(enh_real.shape[2], enh_real.shape[3]),
                               mode="nearest")
        enh_real = (enh_real * ori_up).sum(1, keepdim=True)
        # enh_imag = (enh_imag * ori_up).sum(1, keepdim=True)

        if is_list:
            enh = torch.split(enh_real, [batch_dim, batch_dim], dim=0)

        return enh
