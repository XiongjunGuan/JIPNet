'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:12:25
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-10-31 17:14:57

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_second"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in [
                "channels_last", "channels_second", "channels_first"
        ]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_second":
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            x = x.permute(0, 3, 1, 2)
            return x
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BaseBlock(nn.Module):

    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c,
                               out_channels=dw_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel,
                               out_channels=dw_channel,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel,
                               out_channels=c,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)

        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel,
                      out_channels=dw_channel // 2,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1,
                      bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2,
                      out_channels=dw_channel,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1,
                      bias=True), nn.Sigmoid())

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c,
                               out_channels=ffn_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel,
                               out_channels=c,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)),
                                  requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class EnhanceNet(nn.Module):

    def __init__(self,
                 img_channel=1,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[],
                 dw_expand=1,
                 ffn_expand=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel,
                               out_channels=width,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width,
                      out_channels=img_channel,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      groups=1,
                      bias=True), nn.Sigmoid())

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[
                    BaseBlock(chan, dw_expand, ffn_expand) for _ in range(num)
                ]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[BaseBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                              nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[
                    BaseBlock(chan, dw_expand, ffn_expand) for _ in range(num)
                ]))

        self.padder_size = 2**len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        # for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
        #     x = up(x)
        #     x = x + enc_skip
        #     x = decoder(x)

        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            x = decoder(x)

        x = self.ending(x)
        # x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size -
                     h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size -
                     w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class EnhancerDecoder(nn.Module):

    def __init__(self,
                 img_channel=1,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[],
                 dw_expand=1,
                 ffn_expand=2):
        super().__init__()
        chan = width
        for num in enc_blk_nums:
            chan = chan * 2

        self.middle_blks = nn.ModuleList()
        self.middle_blks = \
            nn.Sequential(
                *[BaseBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                              nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[
                    BaseBlock(chan, dw_expand, ffn_expand) for _ in range(num)
                ]))

        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width,
                      out_channels=img_channel,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      groups=1,
                      bias=True), nn.Sigmoid())

    def forward(self, x):
        x = self.middle_blks(x)

        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            x = decoder(x)

        x = self.ending(x)
        return x


class EnhancerEncoder(nn.Module):

    def __init__(self,
                 img_channel=1,
                 width=16,
                 enc_blk_nums=[],
                 dw_expand=1,
                 ffn_expand=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel,
                               out_channels=width,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()

        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[
                    BaseBlock(chan, dw_expand, ffn_expand) for _ in range(num)
                ]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

    def forward(self, inp):
        x = self.intro(inp)

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            x = down(x)

        return x
