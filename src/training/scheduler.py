"""Learning-rate schedulers used by segmentation training."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer, epochs: int, warmup_epochs: int
) -> LambdaLR:
    """Build a linear warmup followed by cosine learning-rate schedule."""
    if epochs <= 0 or not 0 <= warmup_epochs < epochs:
        raise ValueError(
            "epochs must be positive and warmup_epochs must be below epochs."
        )

    def schedule(epoch: int) -> float:
        """Return the multiplier for one zero-indexed scheduler epoch."""
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=schedule)
