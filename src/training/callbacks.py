"""Training callbacks for checkpointing and early stopping."""

from pathlib import Path

import torch
from torch import nn


class EarlyStopping:
    """Stop training after validation Dice fails to improve for a fixed patience."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        """Record a score and return whether training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model parameters, creating the checkpoint directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
