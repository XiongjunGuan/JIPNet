"""
Description:
Author: Xiongjun Guan
Date: 2024-01-19 22:33:29
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-10-30 17:48:25

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6


class BinaryFocalLoss(nn.Module):
    """
    https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.2, epsilon=1.0e-9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input

        zero_hot_key = 1 - multi_hot_key
        loss = (
            -self.alpha
            * multi_hot_key
            * torch.pow((1 - logits), self.gamma)
            * (logits + self.epsilon).log()
        )
        loss += (
            -(1 - self.alpha)
            * zero_hot_key
            * torch.pow(logits, self.gamma)
            * (1 - logits + self.epsilon).log()
        )
        return loss.mean()


class CompareAlignLoss(nn.Module):
    def __init__(self, w=0.002):
        super().__init__()
        self.focal_loss = BinaryFocalLoss()
        self.w = w

    def forward(
        self,
        cla_pred,
        cla_gt,
        align_pred,
        align_gt,
        lambda_2=0.99,
    ):
        focal_loss = self.focal_loss(
            cla_pred,
            cla_gt,
        )

        pred_b1 = align_pred[:, 0]
        pred_b2 = align_pred[:, 1]
        cosT = torch.div(
            pred_b1, eps + torch.sqrt(torch.square(pred_b1) + torch.square(pred_b2))
        )
        sinT = torch.div(
            pred_b2, eps + torch.sqrt(torch.square(pred_b1) + torch.square(pred_b2))
        )
        pred_norm = torch.cat(
            [
                cosT[:, None],
                sinT[:, None],
                align_pred[:, 2][:, None],
                align_pred[:, 3][:, None],
            ],
            dim=1,
        )

        l2 = torch.square(pred_norm - align_gt)
        l2 = lambda_2 * (l2[:, 0] + l2[:, 1]) + (1 - lambda_2) * (l2[:, 2] + l2[:, 3])

        Lr = torch.sum(l2 * cla_gt.reshape((-1,))) / (torch.sum(cla_gt) + eps)

        loss = focal_loss + self.w * Lr
        items = {
            "focal": focal_loss.item(),
            "Lr": Lr.item(),
        }

        return loss, items
