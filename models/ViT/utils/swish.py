'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:18:03
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-10-31 17:18:14

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)
