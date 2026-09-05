"""Configurable BCE, Dice, and Tversky segmentation losses."""

from torch import Tensor
from torch.nn import functional as F


def soft_dice_loss(logits: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """Compute one minus the batch-averaged soft Dice score."""
    probabilities = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    score = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - score.mean()


def tversky_loss(
    logits: Tensor, target: Tensor, alpha: float = 0.7, beta: float = 0.3
) -> Tensor:
    """Compute one minus the batch-averaged Tversky score."""
    probabilities = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    true_positive = (probabilities * target).sum(dim=(1, 2, 3))
    false_positive = (probabilities * (1 - target)).sum(dim=(1, 2, 3))
    false_negative = ((1 - probabilities) * target).sum(dim=(1, 2, 3))
    score = (true_positive + 1) / (
        true_positive + alpha * false_positive + beta * false_negative + 1
    )
    return 1 - score.mean()


def combined_loss(
    logits: Tensor,
    target: Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.25,
    tversky_weight: float = 0.25,
) -> Tensor:
    """Compute the configured weighted BCE, Dice, and Tversky loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return (
        bce_weight * bce
        + dice_weight * soft_dice_loss(logits, target)
        + tversky_weight * tversky_loss(logits, target)
    )
