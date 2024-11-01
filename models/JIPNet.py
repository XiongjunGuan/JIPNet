'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:09:56
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-01 13:47:16

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch
import torch.nn as nn

from models.Enhancer import EnhancerEncoder
from models.ResNet import ResBlock
from models.ViT.ViT_reg_cla import ViT_reg_cla


class JIPNet(nn.Module):
    # dense sampling verification network
    def __init__(self,
                 input_size=160,
                 img_channel=1,
                 num_classes=1,
                 width=32,
                 enc_blk_nums=[2, 2, 2],
                 dw_expand=1,
                 ffn_expand=2,
                 mid_blk_nums=[4, 6],
                 mid_blk_strides=[1, 2],
                 mid_embed_dims=[512, 768],
                 dec_hidden_dim=256,
                 dec_nhead=8,
                 dec_local_num=6,
                 encoder_pretrain_pth=None):
        super().__init__()

        inner_size = input_size
        chan = width

        # ----- encoder ----- #
        self.encoder = EnhancerEncoder(img_channel=img_channel,
                                       width=width,
                                       enc_blk_nums=enc_blk_nums,
                                       dw_expand=dw_expand,
                                       ffn_expand=ffn_expand)
        for num in enc_blk_nums:
            chan = chan * 2
            inner_size = inner_size // 2

        if encoder_pretrain_pth is not None:
            self.encoder.load_state_dict(
                torch.load(encoder_pretrain_pth, map_location=f'cuda:0'))
            # for p in self.encoder.parameters():
            #     p.requires_grad = False

        # ----- mid layers ----- #
        layers = []
        for num, stride, embed_dim in zip(mid_blk_nums, mid_blk_strides,
                                          mid_embed_dims):
            layers.append(
                nn.Sequential(*[ResBlock(chan, embed_dim, stride=stride)]))
            chan = embed_dim
            inner_size = inner_size // stride
            layers.append(
                nn.Sequential(*[ResBlock(chan, chan) for _ in range(num - 1)]))

        self.mid_layer = nn.Sequential(*layers)

        # ----- decoder ----- #
        local_layer_names = []
        for num in dec_local_num:
            local_layer_names.append(['self', 'cross'] * num)
        self.transformer_layer = ViT_reg_cla(
            embed_dim=chan,
            hidden_dim=dec_hidden_dim,
            num_classes=num_classes,
            nheads=dec_nhead,
            feat_shape=(inner_size, inner_size),
            local_layer_names=local_layer_names,
            attention='linear',
            dp=0.0)

    def forward(self, inps):
        batch_dim = inps[0].shape[0]
        x = torch.cat(inps, dim=0)
        x = self.encoder(x)
        x = self.mid_layer(x)
        [x0, x1] = torch.split(x, [batch_dim, batch_dim], dim=0)

        cla_pred, align_pred = self.transformer_layer(x0, x1)

        cla_pred = torch.sigmoid(cla_pred)

        return cla_pred, align_pred
