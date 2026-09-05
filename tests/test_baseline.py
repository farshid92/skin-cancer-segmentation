"""Tests for the first segmentation baseline."""

import numpy as np
import torch

from src.data.preprocessing import prepare_image, prepare_mask
from src.evaluation.metrics import dice_score, iou_score
from src.losses.combined import combined_loss
from src.models.baseline import BaselineSegmenter
from src.utils.seed import set_seed


def test_preprocessing_preserves_tensor_contract() -> None:
    """Prepared image and mask tensors use the documented layout and dtype."""
    image = np.full((12, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:8, 3:9] = 255

    prepared_image = prepare_image(image, 8)
    prepared_mask = prepare_mask(mask, 8)

    assert prepared_image.shape == (3, 8, 8)
    assert prepared_image.dtype == torch.float32
    assert prepared_mask.shape == (1, 8, 8)
    assert set(prepared_mask.unique().tolist()) <= {0.0, 1.0}


def test_baseline_supports_forward_and_backward() -> None:
    """The baseline emits logits and a finite trainable loss."""
    set_seed(42)
    model = BaselineSegmenter()
    images = torch.randn(2, 3, 16, 16)
    masks = torch.randint(0, 2, (2, 1, 16, 16)).float()

    logits = model(images)
    loss = combined_loss(logits, masks)
    loss.backward()

    assert logits.shape == masks.shape
    assert torch.isfinite(loss)
    assert torch.isfinite(dice_score(logits, masks)).all()
    assert torch.isfinite(iou_score(logits, masks)).all()


def test_empty_masks_have_perfect_dice() -> None:
    """Two empty masks receive the specified perfect Dice score."""
    logits = torch.full((1, 1, 4, 4), -20.0)
    target = torch.zeros_like(logits)

    assert dice_score(logits, target).item() == 1.0
    assert iou_score(logits, target).item() == 1.0
