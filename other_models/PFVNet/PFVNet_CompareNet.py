'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 10:02:35
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 10:02:51

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import torch
from torch import nn
from torch.nn import functional as F


def get_patch(input_tensor, k, output_h, output_w):
    n, _, input_h, input_w = input_tensor.shape
    stride_h = (input_h - output_h) // (k - 1)
    stride_w = (input_w - output_w) // (k - 1)

    output_tensor = []
    for i in range(k):
        for j in range(k):
            window = input_tensor[:, :, i * stride_h:i * stride_h + output_h,
                                  j * stride_w:j * stride_w + output_w]
            output_tensor.append(window)

    output_tensor = torch.cat(output_tensor, dim=0)

    return output_tensor, k * k


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        # 通过stride减少参数维度
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in,
                               ch_out,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,
                               ch_out,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # [b, ch_in, h, w] => [b, ch_out, h, w]
        if stride != 1 or ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out))
        else:
            self.extra = nn.Sequential()

    def forward(self, x):
        '''
        :param x: [b, ch, h, w]
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return F.relu(out)


class ProductBlock(nn.Module):

    def __init__(self, ch_in, stride=1):
        # 通过stride减少参数维度
        super(ProductBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=3,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=5,
                      padding=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(ch_in * 3,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_in,
                      kernel_size=3,
                      stride=stride,
                      dilation=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
        )

    def forward(self, x):
        '''
        :param x: [b, ch, h, w]
        :return:
        '''
        x1 = self.conv4(self.conv1(x))
        x2 = self.conv5(self.conv2(x))
        x3 = self.conv6(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = torch.sigmoid(x)

        return x


class CompareEncoder(nn.Module):

    def __init__(self, out_ch=512):
        super(CompareEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # follow 4 blocks
        # [b, 64, h, w] => [b, 128, h/2, w/2]
        self.layer1 = ResBlk(64, 128, stride=2)
        self.product_layer = ProductBlock(128)
        # [b, 128, h/2, w/2] => [b, 256, h/4, w/4]
        self.layer2 = ResBlk(128, 256, stride=2)
        # [b, 256, h/4, w/4] => [b, 512, h/8, w/8]
        self.layer3 = ResBlk(256, 512, stride=2)
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
        self.layer4 = ResBlk(512, 512, stride=2)

        self.out_layer = nn.Sequential(
            nn.Linear(512 * 1 * 1, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):

        # [b, 3, h, w] => [b, 64, h, w]
        x = self.conv1(x)

        # [b, 64, h, w] => [b, 512, h/16, w/16]
        x = self.layer1(x)

        x = x * self.product_layer(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b, 512, h/16, w/16] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])

        # [b, 512, 1, 1] => [b, 512]
        x = torch.flatten(x, 1)
        # [b, 512] => [b, 10]
        x = self.out_layer(x)

        return x


class CompareNet(nn.Module):

    def __init__(self, p1=3, p2=2, pw1=64, pw2=96):
        super(CompareNet, self).__init__()
        self.p1 = p1
        self.pw1 = pw1
        self.p2 = p2
        self.pw2 = pw2

        self.encoder1 = CompareEncoder(out_ch=512)
        self.encoder2 = CompareEncoder(out_ch=512)
        self.encoder3 = CompareEncoder(out_ch=512)

        self.mlp1 = nn.Sequential(nn.Linear(512, 1))
        self.mlp2 = nn.Sequential(nn.Linear(1536, 512), nn.BatchNorm1d(512),
                                  nn.ReLU(True), nn.Linear(512, 1))

    def forward(self, inputs):
        b = inputs.shape[0]

        x1, n1 = get_patch(inputs.clone(), self.p1, self.pw1, self.pw1)
        x2, n2 = get_patch(inputs.clone(), self.p2, self.pw2, self.pw2)
        x3, n3 = inputs.clone(), 1

        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x3 = self.encoder3(x3)

        y1 = self.mlp1(x1)
        y2 = self.mlp1(x2)
        y3 = self.mlp1(x3)

        y1 = torch.sigmoid(y1)
        y2 = torch.sigmoid(y2)
        y3 = torch.sigmoid(y3)

        # (b*n,c) -> [(b,c,1)] * n
        x1 = torch.split(x1.unsqueeze(2), [b] * n1, dim=0)
        x2 = torch.split(x2.unsqueeze(2), [b] * n2, dim=0)
        x3 = torch.split(x3.unsqueeze(2), [b] * n3, dim=0)

        # [(b,c,1)] * n -> (b,c,n)
        x1 = torch.cat(x1, dim=2)
        x2 = torch.cat(x2, dim=2)
        x3 = torch.cat(x3, dim=2)

        # (b,c,n) -> (b,c)
        x1 = F.max_pool1d(x1, kernel_size=n1).squeeze(2)
        x2 = F.max_pool1d(x2, kernel_size=n2).squeeze(2)
        x3 = F.max_pool1d(x3, kernel_size=n3).squeeze(2)

        # (b,c) -> (b,3c)
        x = torch.cat([x1, x2, x3], dim=1)

        y = self.mlp2(x)
        y = torch.sigmoid(y)

        return y, y1, y2, y3


if __name__ == '__main__':
    comparenet = CompareNet()

    x1 = torch.randn(10, 25, 2, 64, 64)
    x2 = torch.randn(10, 9, 2, 96, 96)
    x3 = torch.randn(10, 1, 2, 192, 192)

    print(comparenet(x1, x2, x3).shape)
