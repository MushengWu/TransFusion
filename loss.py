from typing import List

import torch
import torch.nn as nn


def DiceLoss(image: torch.Tensor, mask: torch.Tensor, eps=1e-5, Sigmoid=True):
    reduce_axis: List[int] = torch.arange(2, len(image.shape)).tolist()

    if Sigmoid:
        image = torch.sigmoid(image)
    inter = torch.sum(image * mask, dim=reduce_axis)
    union = torch.sum(image, dim=reduce_axis) + torch.sum(mask, dim=reduce_axis)

    dice = (2 * inter + eps) / (union + eps)
    loss = torch.mean(1.0 - dice)

    return loss


class DiceCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', weight_dice=1.0, weight_ce=1.0, sigmoid=False):
        super(DiceCELoss, self).__init__()
        self.CE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        self.sigmoid = sigmoid
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        ce = self.CE(image, mask.float())
        dice = DiceLoss(image, mask, Sigmoid=self.sigmoid)

        return self.weight_ce * ce + self.weight_dice * dice
