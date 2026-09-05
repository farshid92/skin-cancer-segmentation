"""SMP U-Net++ model builder."""

from collections.abc import Sequence
from typing import cast

import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
from torch import nn


def decoder_channels(scale: int) -> tuple[int, int, int, int, int]:
    """Return the five U-Net++ decoder widths for a configured base scale."""
    if scale not in {16, 32, 64}:
        raise ValueError("decoder_channels_scale must be one of: 16, 32, 64.")
    return (scale * 8, scale * 4, scale * 2, scale, scale // 2)


def build_unetpp(
    encoder: str,
    encoder_weights: str | None,
    decoder_channels_scale: int,
) -> nn.Module:
    """Build U-Net++ configured to return one segmentation logit per pixel."""
    channels: Sequence[int] = decoder_channels(decoder_channels_scale)
    return cast(
        nn.Module,
        smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            decoder_channels=channels,
            in_channels=3,
            classes=1,
            activation=None,
        ),
    )
