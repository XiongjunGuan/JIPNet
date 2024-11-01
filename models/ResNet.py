'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:14:24
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-01 13:47:22

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, expansion=2):
        # 通过stride减少参数维度
        super().__init__()

        mid_channel = out_channel // expansion
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=mid_channel,
            kernel_size=1,
            stride=1,
        )  # squeeze channels
        self.bn1 = nn.BatchNorm2d(mid_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=mid_channel,
                               out_channels=mid_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=mid_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
        )  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        # [b, ch_in, h, w] => [b, ch_out, h, w]
        if stride != 1 or in_channel != out_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=1,
                          stride=stride), nn.BatchNorm2d(out_channel))
        else:
            self.extra = nn.Identity()

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        :param x: [b, ch, h, w]
        :return:
        '''
        identity = x
        identity = self.extra(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
