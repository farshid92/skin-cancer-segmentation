"""Single entry point for configured segmentation models."""

from typing import cast

from omegaconf import DictConfig
from torch import nn

from src.models.unetpp import build_unetpp


def build_model(config: DictConfig) -> nn.Module:
    """Build a segmentation model from the selected model configuration."""
    model_name = cast(str, config.name).lower()
    if model_name != "unetpp":
        raise ValueError(f"Unsupported model: {model_name}")
    return build_unetpp(
        encoder=cast(str, config.encoder),
        encoder_weights=cast(str | None, config.get("encoder_weights")),
        decoder_channels_scale=cast(int, config.decoder_channels_scale),
    )
