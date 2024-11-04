'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 10:59:20
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 11:10:22

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inception import *


class DeepPrint_stn(nn.Module):

    def __init__(self,
                 num_in=1,
                 ndim_feat=96,
                 num_classes=24000,
                 crop_shape=np.array([285, 285]),
                 pretrained=False) -> None:
        super().__init__()
        self.num_in = num_in  # number of input channel
        self.ndim_feat = ndim_feat  # number of dimension of latent feature
        self.num_classes = num_classes  # number of classes
        self.crop_shape = crop_shape  # cropping shape size
        # input size=(299, 299)

        # fingerprint pose
        self.localization = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(True),
            nn.Linear(64, 3),
        )

        # stem module
        self.stem = nn.Sequential(
            BasicConv2d(1, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
        )  # size=(35, 35)

        # texture component
        self.texture = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
        )  # size=(8, 8)

        # minutiae component
        self.minutiae_e = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
        )  # size=(35, 35)
        self.minutiae_d = nn.Sequential(
            BasicDeConv2d(384,
                          128,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          outpadding=1),
            BasicConv2d(128, 128, kernel_size=7, stride=1),
            BasicDeConv2d(128,
                          32,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          outpadding=1),
            nn.Conv2d(32, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),  # after 20221215
        )  # size=(128, 128)

        self.texture_f = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout2d(0.2),
            nn.Linear(1536, self.ndim_feat),
        )
        self.texture_p = nn.Linear(self.ndim_feat, num_classes)

        self.minutiae_f = nn.Sequential(
            BasicConv2d(384, 768, kernel_size=3, stride=1, padding=1),
            BasicConv2d(768, 768, kernel_size=3, stride=2, padding=1),
            BasicConv2d(768, 896, kernel_size=3, stride=1, padding=1),
            BasicConv2d(896, 1024, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout2d(0.2),
            nn.Linear(1024, self.ndim_feat),
        )
        self.minutiae_p = nn.Linear(self.ndim_feat, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.localization[-1].weight.data.zero_()

    def stn(self, input, align=False):
        B, C, H, W = input.size()

        if align:
            pose = torch.zeros([B, 3]).type_as(input)
        else:
            input_loc = F.interpolate(input, (128, 128),
                                      mode="bilinear",
                                      align_corners=False)
            z = self.localization(input_loc.detach())
            pose = z.clamp(-1, 1)

        cos_theta = torch.deg2rad(pose[:, 2] * 60).cos()
        sin_theta = torch.deg2rad(pose[:, 2] * 60).sin()

        # normal size, for fingernet (minutiae map prediction)
        scale_x = (self.crop_shape[1] - 1) * 1.0 / (W - 1)
        scale_y = (self.crop_shape[0] - 1) * 1.0 / (H - 1)
        T = torch.stack(
            (
                torch.stack(
                    [scale_x * cos_theta, scale_x * sin_theta, pose[:, 0]],
                    dim=1),
                torch.stack(
                    [scale_y * -sin_theta, scale_y * cos_theta, pose[:, 1]],
                    dim=1),
            ),
            dim=1,
        )
        grid = F.affine_grid(T,
                             torch.Size((B, C, *self.crop_shape)),
                             align_corners=False)
        y = F.grid_sample(input, grid, mode="bilinear", align_corners=False)

        return y, pose, T

    def get_embedding(self, input, align=False):

        input_stn, pose, _ = self.stn(input, align)

        # if len(time_lst) < 20:
        #     time_lst.append(time.time() - tic)
        # else:
        #     print(np.mean(time_lst))

        x = F.interpolate(input_stn, (299, 299),
                          mode="bilinear",
                          align_corners=False)
        x = self.stem(x)

        texture = self.texture(x)
        texture_f = self.texture_f(texture)

        minutiae_e = self.minutiae_e(x)
        minutiae_f = self.minutiae_f(minutiae_e)

        return torch.cat((F.normalize(texture_f), F.normalize(minutiae_f)),
                         dim=1), pose

    def forward(self, input):
        input_stn, pose, T = self.stn(input)

        x = F.interpolate(input_stn, (299, 299),
                          mode="bilinear",
                          align_corners=True)
        x = self.stem(x)

        texture = self.texture(x)
        texture_f = self.texture_f(texture)
        texture_p = self.texture_p(texture_f)

        minutiae_e = self.minutiae_e(x)
        minutiae_d = self.minutiae_d(minutiae_e)
        minutiae_f = self.minutiae_f(minutiae_e)
        minutiae_p = self.minutiae_p(minutiae_f)

        return {
            "text_f": texture_f,
            "minu_f": minutiae_f,
            "text_p": texture_p,
            "minu_p": minutiae_p,
            "text_label": torch.argmax(texture_p, dim=1).detach(),
            "minu_label": torch.argmax(minutiae_p, dim=1).detach(),
            "minu_map": minutiae_d,
            "minu_lst": torch.split(minutiae_d.detach(), 3, dim=1),
            "aligned": input_stn,
            "pose": pose,
        }
