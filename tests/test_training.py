"""Tests for the training loop, scheduling, and callback behavior."""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.models.baseline import BaselineSegmenter
from src.training.callbacks import EarlyStopping
from src.training.scheduler import build_warmup_cosine_scheduler
from src.training.trainer import Trainer


class FakeClient:
    """Capture metrics without requiring an MLflow server in tests."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, str, float, int]] = []

    def log_metric(self, run_id: str, key: str, value: float, step: int) -> None:
        """Record a metric call for assertion."""
        self.metrics.append((run_id, key, value, step))


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    """Provide deterministic synthetic segmentation batches for trainer tests."""

    def __len__(self) -> int:
        """Return the fixed fixture size."""
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one tiny RGB image and binary lesion mask."""
        return {"image": torch.ones(3, 16, 16), "mask": torch.zeros(1, 16, 16)}


def test_trainer_logs_metrics_and_saves_best_checkpoint(tmp_path: Path) -> None:
    """Trainer completes an epoch and centralizes metrics through its client."""
    model = BaselineSegmenter()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=2, warmup_epochs=1)
    client = FakeClient()
    loader = DataLoader(TinyDataset(), batch_size=1)
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        client,
        "run",
        tmp_path / "best.pt",
        torch.device("cpu"),
        2,
        False,
        1.0,
        1,
    )

    trainer.fit(loader, loader, epochs=2)

    assert (tmp_path / "best.pt").is_file()
    assert {metric[1] for metric in client.metrics} >= {
        "train_loss",
        "val_dsc",
        "val_iou",
    }


def test_early_stopping_respects_patience() -> None:
    """Early stopping activates after the configured number of non-improvements."""
    callback = EarlyStopping(patience=2)

    assert not callback.update(0.5)
    assert not callback.update(0.5)
    assert callback.update(0.5)
