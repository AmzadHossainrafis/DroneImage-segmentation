import torch
import torch.nn as nn
import numpy as np


# funtion iou
class IOU(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, pred, target):
        # intersection between pred and target at dimention -1
        intersection = np.logical_and(target, pred)
        # union between pred and target at dimention -1
        union = np.logical_or(target, pred)
        # sum of intersection and union
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def __str__(self):
        return f"IOU(n_classes={self.n_classes})"


class Dice(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, pred, target):
        # intersection between pred and target at dimention -1
        intersection = np.logical_and(target, pred)
        # union between pred and target at dimention -1
        union = np.logical_or(target, pred)
        # sum of intersection and union
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def __str__(self):
        return f"Dice(n_classes={self.n_classes})"
