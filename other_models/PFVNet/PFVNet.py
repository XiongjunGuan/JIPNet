'''
Description: 
Author: Xiongjun Guan
Date: 2024-01-21 00:11:28
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 10:06:25

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from other_models.PFVNet.PFVNet_AlignNet import AlignNet
from other_models.PFVNet.PFVNet_CompareNet import CompareNet
from other_models.PFVNet.utils import AffinePatch


class PFVNet(nn.Module):

    def __init__(self,
                 p1=3,
                 p2=2,
                 pw1=64,
                 pw2=96,
                 align_pth=None,
                 compare_pth=None):
        super(PFVNet, self).__init__()
        self.align_net = AlignNet()
        self.compare_net = CompareNet(p1=p1, p2=p2, pw1=pw1, pw2=pw2)
        self.affine_func = AffinePatch()

        if align_pth is not None:
            self.align_net.load_state_dict(torch.load(align_pth))
        if compare_pth is not None:
            self.compare_net.load_state_dict(torch.load(compare_pth))

    def forward(self, inputs):
        img1 = inputs[0]
        img2 = inputs[1]
        align_pred = self.align_net(inputs)
        img2 = self.affine_func(img2, align_pred)
        y, y1, y2, y3 = self.compare_net(torch.cat((img1, img2), dim=1))
        return y, y1, y2, y3, align_pred
