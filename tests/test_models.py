"""Tests for configured segmentation models."""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.factory import build_model
from src.models.unetpp import decoder_channels


def test_unetpp_factory_returns_single_channel_logits() -> None:
    """Factory-built U-Net++ preserves batch and spatial dimensions."""
    config = OmegaConf.create(
        {
            "name": "unetpp",
            "encoder": "resnet18",
            "encoder_weights": None,
            "decoder_channels_scale": 16,
        }
    )
    model = build_model(config)

    logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 1, 64, 64)


def test_decoder_channels_rejects_unknown_scale() -> None:
    """Unsupported decoder scales fail before model construction."""
    with pytest.raises(ValueError, match="decoder_channels_scale"):
        decoder_channels(24)


def test_factory_rejects_unknown_model() -> None:
    """Unsupported model names fail with an actionable exception."""
    config = OmegaConf.create({"name": "unknown"})

    with pytest.raises(ValueError, match="Unsupported model"):
        build_model(config)
