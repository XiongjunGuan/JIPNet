'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:23:40
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-01 15:58:11

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
"""
The prototype of transformer block can refer to a https://github.com/zju3dv/LoFTR
"""

import torch
import torch.nn as nn
from einops.einops import rearrange
from timm.models.layers import trunc_normal_

from models.ViT.module.transformer import LocalFeatureTransformer
from models.ViT.utils.merge import PatchMerging
from models.ViT.utils.position_encoding import PositionEncodingSine
from models.ViT.utils.swish import MemoryEfficientSwish


class ClassificationHead(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_classes, dp=0.):
        super().__init__()
        # Classifier head
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            MemoryEfficientSwish(),
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            MemoryEfficientSwish(),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dp) if dp > 0 else nn.Identity()
        self.head = nn.Sequential(nn.Linear(hidden_dim, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, L, C = x.shape
        x = self.att_mlp(x.reshape(-1, C)).reshape(B, L, -1)
        x = rearrange(x, "B L C -> B C L")
        x = self.avgpool(x).squeeze()
        x = self.drop(x)
        x = self.head(x)
        return x


class AlignHead(nn.Module):

    def __init__(self, embed_dim, hidden_dim=512, num_classes=4, dp=0.):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            MemoryEfficientSwish(),
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            MemoryEfficientSwish(),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dp) if dp > 0 else nn.Identity()
        self.head = nn.Sequential(nn.Linear(hidden_dim, num_classes))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, L, C = x.shape
        x = self.att_mlp(x.reshape(-1, C)).reshape(B, L, -1)
        x = rearrange(x, "B L C -> B C L")
        x = self.avgpool(x).squeeze()
        x = self.drop(x)
        x = self.head(x)
        return x


class ViT_reg_cla(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 hidden_dim=256,
                 num_classes=1,
                 nheads=[6, 12],
                 feat_shape=(10, 10),
                 local_layer_names=[['self', 'cross'] * 6,
                                    ['self', 'cross'] * 2],
                 attention='linear',
                 dp=0.):
        super().__init__()
        self.feat_shape = feat_shape
        self.num_patches = feat_shape[0] * feat_shape[1]
        # Modules
        self.pos_encoding1 = PositionEncodingSine(embed_dim, temp_bug_fix=True)

        self.local_transformer1 = LocalFeatureTransformer(
            d_model=embed_dim,
            nhead=nheads[0],
            layer_names=local_layer_names[0],
            attention=attention)

        self.patch_merging = PatchMerging(dim=embed_dim, out_dim=hidden_dim)
        embed_dim = hidden_dim

        self.pos_encoding2 = PositionEncodingSine(embed_dim, temp_bug_fix=True)

        self.local_transformer_cla = LocalFeatureTransformer(
            d_model=embed_dim,
            nhead=nheads[1],
            layer_names=local_layer_names[1],
            attention=attention)

        self.local_transformer_align = LocalFeatureTransformer(
            d_model=embed_dim,
            nhead=nheads[1],
            layer_names=local_layer_names[1],
            attention=attention)

        self.cla_head = ClassificationHead(embed_dim=embed_dim * 2,
                                           hidden_dim=hidden_dim,
                                           num_classes=num_classes,
                                           dp=dp)
        self.align_head = AlignHead(embed_dim=embed_dim * 2,
                                    hidden_dim=embed_dim)

    def forward(self, feat0, feat1):
        """ 
            'feat0': (torch.Tensor): (B, C, H, W)
            'feat1': (torch.Tensor): (B, C, H, W)
            'mask0'(optional) : (torch.Tensor): (B, H, W) '0' indicates a padded position
            'mask1'(optional) : (torch.Tensor): (B, H, W)
        """
        B, C, H, W = feat0.shape

        # add featmap with positional encoding, then flatten it to sequence [B, HW, C]
        feat0 = rearrange(self.pos_encoding1(feat0), 'n c h w -> n (h w) c')
        feat1 = rearrange(self.pos_encoding1(feat1), 'n c h w -> n (h w) c')

        mask0 = mask1 = None  # mask is useful in training

        feat0, feat1 = self.local_transformer1(feat0, feat1, mask0, mask1)

        feat0 = self.patch_merging(feat0, H, W)
        feat1 = self.patch_merging(feat1, H, W)
        H, W = H // 2, W // 2

        B, L, C = feat0.shape
        feat0 = rearrange(
            self.pos_encoding2(feat0.transpose(2, 1).view(B, C, H, W)),
            'n c h w -> n (h w) c')
        feat1 = rearrange(
            self.pos_encoding2(feat1.transpose(2, 1).view(B, C, H, W)),
            'n c h w -> n (h w) c')

        feat0_cla, feat1_cla = self.local_transformer_cla(
            feat0, feat1, mask0, mask1)
        cla_feat = torch.cat([feat0_cla, feat1_cla], dim=2)
        cla_pred = self.cla_head(cla_feat)

        feat0_align, feat1_align = self.local_transformer_align(
            feat0, feat1, mask0, mask1)
        align_feat = torch.cat([feat0_align, feat1_align], dim=2)
        align_pred = self.align_head(align_feat)

        return cla_pred, align_pred
