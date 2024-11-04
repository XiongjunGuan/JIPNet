'''
Description: 
Author: Xiongjun Guan
Date: 2024-11-04 11:17:53
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-04 11:21:57

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from . import resnet


class NormalizeModule(nn.Module):

    def __init__(self, m0=0.0, var0=1.0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m)**2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


# STN for afrnet
class STNnet(nn.Module):

    def __init__(self, input_size=224, num_in=3):
        super(STNnet, self).__init__()
        # input size is 224
        last_size = input_size // 32
        self.localization = nn.Sequential(
            nn.Conv2d(num_in, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2),
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
            nn.Linear(64 * last_size * last_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 4),  # do not use the scale
        )

    def stn(self, input, align=False):
        B, C, H, W = input.size()

        if align:
            pose = torch.zeros([B, 3]).type_as(input)
        else:
            # input_loc = F.interpolate(input, (128, 128), mode="bilinear", align_corners=False)
            z = self.localization(input.detach())
            s, z = z[:, 0], z[:, 1:]
            s.clamp_(0.8, 1.2)  # scale limitation
            pose = z.clamp(-1, 1)  # scale limitation
            theta, tx, ty = pose[:, 0] * 60, pose[:, 1], pose[:, 2]
            pose = torch.stack([s, tx, ty, theta], dim=1)

        cos_theta = torch.deg2rad(theta).cos()
        sin_theta = torch.deg2rad(theta).sin()

        # normal size, for fingernet (minutiae map prediction)

        T = torch.stack(
            (
                torch.stack([s * cos_theta, s * sin_theta, tx], dim=1),
                torch.stack([s * -sin_theta, s * cos_theta, ty], dim=1),
            ),
            dim=1,
        )
        grid = F.affine_grid(T, torch.Size((B, C, H, W)), align_corners=False
                             )  # apply affine transformation on the same size
        y = F.grid_sample(input,
                          grid,
                          mode="bilinear",
                          align_corners=False,
                          padding_mode="border")

        return y, pose, T

    def forward(self, input, align=False):
        input_aligned, pose, T = self.stn(input, align)
        return input_aligned, pose, T


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class AttentionLayer(nn.Module):

    def __init__(self, input_size=224):
        super(AttentionLayer, self).__init__()
        inner_size = input_size // 16
        self.num_patches = inner_size * inner_size
        self.att_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(True),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1,
                                                  384),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(384, 6, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(12)
        ])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.vit_feature = nn.Linear(384, 384)
        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.num_patches**.5),
                                            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward(self, inner_feature):
        B, C, H, W = inner_feature.shape
        # inner_feature in B x 1024 x 14 x 14
        inner_feature = inner_feature.flatten(2).transpose(1, 2)
        # import pdb;pdb.set_trace()
        # after the MLP further embedding
        inner_feature = self.att_mlp(inner_feature.reshape(-1, C)).reshape(
            B, H * W, -1)  # N, L, D
        # inner_feature = inner_feature.transpose(1, 2)
        inner_feature = inner_feature + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(inner_feature.shape[0], -1, -1)
        inner_feature = torch.cat((cls_tokens, inner_feature),
                                  dim=1)  # N, L+1, D
        for blk in self.blocks:
            inner_feature = blk(inner_feature)
        # exclude the cls token
        inner_feature = inner_feature[:, 1:, :]
        inner_feature = inner_feature.reshape(B, H * W, -1).transpose(1, 2)
        inner_feature = self.avgpool(inner_feature).squeeze(-1)
        vit_feature = self.vit_feature(inner_feature)
        return vit_feature


class AFRNet(nn.Module):

    def __init__(self,
                 input_size=224,
                 pretrained=True,
                 num_classes=384,
                 is_stn=False):
        super().__init__()
        self.input_norm = NormalizeModule(m0=0.0, var0=1.0, eps=1e-6)
        self.is_stn = is_stn
        if self.is_stn:
            self.stn = STNnet(input_size)
        block = resnet.Bottleneck
        layers = [3, 4, 6, 3]
        num_layers = [64, 128, 256, 512]  # fixed
        self.backbone_model = resnet._resnet('resnet50',
                                             block,
                                             layers,
                                             pretrained,
                                             True,
                                             num_layers=num_layers,
                                             num_in=3,
                                             num_classes=num_classes)
        # base_layers = list(base_model.children())
        self.vit_head = AttentionLayer(input_size)

    def get_embedding(self, x):
        x = self.input_norm(x)
        if self.is_stn:
            x, pose, T = self.stn(x)
        cnn_feature, inner_feature = self.backbone_model(x)
        vit_feature = self.vit_head(inner_feature)
        return cnn_feature, vit_feature

    def forward(self, x):
        x = self.input_norm(x)  # normalize the input
        if self.is_stn:
            x, pose, T = self.stn(x)
        # test the
        cnn_feature, inner_feature = self.backbone_model(x)
        vit_feature = self.vit_head(inner_feature)
        return cnn_feature, vit_feature
