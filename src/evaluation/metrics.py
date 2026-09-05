"""Pixel-wise segmentation metrics."""

import torch
from torch import Tensor


def dice_score(logits: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute per-image Dice scores with explicit empty-mask handling."""
    prediction = logits.sigmoid() >= threshold
    truth = target >= threshold
    intersection = (prediction & truth).sum(dim=(1, 2, 3)).float()
    predicted = prediction.sum(dim=(1, 2, 3))
    actual = truth.sum(dim=(1, 2, 3))
    denominator = predicted + actual
    score = (2 * intersection) / denominator.clamp_min(1)
    both_empty = denominator == 0
    return torch.where(both_empty, torch.ones_like(score), score)


def iou_score(logits: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute per-image intersection-over-union scores."""
    prediction = logits.sigmoid() >= threshold
    truth = target >= threshold
    intersection = (prediction & truth).sum(dim=(1, 2, 3)).float()
    union = (prediction | truth).sum(dim=(1, 2, 3)).float()
    return torch.where(union == 0, torch.ones_like(union), intersection / union)
