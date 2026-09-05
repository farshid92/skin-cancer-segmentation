"""Deterministic image and mask preparation for segmentation."""

from typing import Final

import cv2
import numpy as np
import torch

IMAGE_MEAN: Final[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD: Final[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def remove_hair(image: np.ndarray, kernel_size: int = 17) -> np.ndarray:
    """Remove dark hair-like structures using black-hat morphology and inpainting."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [H, W, 3].")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)


def prepare_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize and ImageNet-normalize an RGB image into a float tensor."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [H, W, 3].")
    resized = cv2.resize(
        image, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    values = resized.astype(np.float32) / 255.0
    normalized = (values - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float()


def prepare_mask(mask: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize and binarize a lesion mask into a float tensor."""
    if mask.ndim not in (2, 3):
        raise ValueError("Expected a mask with shape [H, W] or [H, W, 1].")
    mask_2d = mask[..., 0] if mask.ndim == 3 else mask
    resized = cv2.resize(
        mask_2d, (image_size, image_size), interpolation=cv2.INTER_NEAREST
    )
    binary = (resized.astype(np.float32) > 0.5).astype(np.float32)
    return torch.from_numpy(binary[None, ...].copy()).float()
