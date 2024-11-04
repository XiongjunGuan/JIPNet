'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 10:01:39
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 10:01:48

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class basic_block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = nn.Sequential()

    def forward(self, inx):
        x = self.relu(self.bn1(self.conv1(inx)))
        x = self.bn2(self.conv2(x))
        out = x + self.downsample(inx)
        return F.relu(out)


class Resnet34(nn.Module):

    def __init__(self,
                 basicBlock=basic_block,
                 blockNums=[3, 4, 6, 3],
                 nb_classes=256):
        super(Resnet34, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2,
                               self.in_planes,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(basicBlock, blockNums[0], 64, 1)
        self.layer2 = self._make_layers(basicBlock, blockNums[1], 128, 2)
        self.layer3 = self._make_layers(basicBlock, blockNums[2], 256, 2)
        self.layer4 = self._make_layers(basicBlock, blockNums[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, nb_classes)
        # self.softmax = nn.Softmax(dim=1)

    def _make_layers(self, basicBlock, blockNum, plane, stride):
        layers = []
        for i in range(blockNum):
            if i == 0:
                layer = basicBlock(self.in_planes, plane, 3, stride=stride)
            else:
                layer = basicBlock(plane, plane, 3, stride=1)
            layers.append(layer)
        self.in_planes = plane
        return nn.Sequential(*layers)

    def forward(self, inx):
        x = self.maxpool(self.relu(self.bn1(self.conv1(inx))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out


class AlignNet(nn.Module):

    def __init__(self):
        super(AlignNet, self).__init__()
        self.res34 = Resnet34(nb_classes=256)
        self.mlp = nn.Sequential(nn.Linear(1024, 4))

    def forward(self, inputs):
        img1 = inputs[0]
        img2 = inputs[1]
        img2_90 = torch.rot90(img2, k=1, dims=(2, 3))
        img2_180 = torch.rot90(img2, k=2, dims=(2, 3))
        img2_270 = torch.rot90(img2, k=3, dims=(2, 3))
        input1 = torch.cat([img1, img2], dim=1)
        input2 = torch.cat([img1, img2_180], dim=1)
        input3 = torch.cat([img1, img2_90], dim=1)
        input4 = torch.cat([img1, img2_270], dim=1)
        feat_1 = self.res34(input1)
        feat_2 = self.res34(input2)
        feat_3 = self.res34(input3)
        feat_4 = self.res34(input4)

        feat = torch.cat([feat_1, feat_2, feat_3, feat_4], dim=1)
        out = self.mlp(feat)

        return out
