'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 10:38:14
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 10:44:17

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3,
                                            padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class Unet_down(nn.Module):

    def __init__(self):
        super(Unet_down, self).__init__()
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 512)
        self.fc = nn.Linear(512 * 12 * 12, 512)
        self.bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        z = x  # for estimate pd
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)

        y = x  # feature
        y = self.dropout(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        y = self.bn(y)

        y = F.normalize(y, p=2, dim=1)

        return z, y


class Unet_up(nn.Module):

    def __init__(self):
        super(Unet_up, self).__init__()
        self.up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv(128, 128)
        self.conv10 = nn.Conv2d(128, 2, 1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.up6(x)
        x = self.conv6(x)
        x = self.conv10(x)
        out = nn.Tanh()(x)
        return out


class DesNet(nn.Module):

    def __init__(self):
        super(DesNet, self).__init__()
        self.embedding_net = Unet_down()
        self.regreesion_net = Unet_up()

    def forward(self, x1, x2):
        c1, feature1 = self.embedding_net(x1)
        c2, feature2 = self.embedding_net(x2)
        output1 = self.regreesion_net(c1)
        output2 = self.regreesion_net(c2)

        return feature1, feature2, output1, output2

    def get_embedding(self, x):
        c, feature = self.embedding_net(x)
        return feature
