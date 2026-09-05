"""Small convolutional baseline used for fast smoke tests."""

from typing import cast

from torch import Tensor, nn


class BaselineSegmenter(nn.Module):
    """Predict a one-channel segmentation logit map from RGB images."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return segmentation logits with shape [B, 1, H, W]."""
        return cast(Tensor, self.head(self.encoder(inputs)))
