"""AMP-enabled model training with centralized MLflow metric logging."""

from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.evaluation.metrics import dice_score, iou_score
from src.losses.combined import combined_loss
from src.training.callbacks import EarlyStopping, save_checkpoint


class MetricClient(Protocol):
    """Minimal MLflow client interface consumed by the trainer."""

    def log_metric(self, run_id: str, key: str, value: float, step: int) -> None:
        """Log one scalar metric for an experiment run."""


class Trainer:
    """Train one segmentation model and retain its best validation checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        client: MetricClient,
        run_id: str,
        checkpoint_path: Path,
        device: torch.device,
        patience: int,
        amp_enabled: bool,
        gradient_clip_norm: float,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.client = client
        self.run_id = run_id
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.stopper = EarlyStopping(patience)
        self.gradient_clip_norm = gradient_clip_norm
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and device.type == "cuda"
        )

    def fit(
        self,
        train_loader: DataLoader[dict[str, Tensor]],
        validation_loader: DataLoader[dict[str, Tensor]],
        epochs: int,
    ) -> float:
        """Train for up to epochs and return the best validation Dice score."""
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, training=True)[0]
            validation_loss, validation_dice, validation_iou = self._run_epoch(
                validation_loader, training=False
            )
            self._log_epoch(
                epoch, train_loss, validation_loss, validation_dice, validation_iou
            )
            self.scheduler.step()
            if validation_dice >= self.stopper.best_score:
                save_checkpoint(self.model, self.checkpoint_path)
            if self.stopper.update(validation_dice):
                break
        return self.stopper.best_score

    def _run_epoch(
        self, loader: DataLoader[dict[str, Tensor]], training: bool
    ) -> tuple[float, float, float]:
        """Run one train or validation epoch and return loss, Dice, and IoU means."""
        self.model.train(training)
        totals = torch.zeros(3, device=self.device)
        with torch.set_grad_enabled(training):
            for batch in loader:
                loss, dice, iou = self._step(batch, training)
                totals += torch.tensor([loss, dice, iou], device=self.device)
        averages = (totals / len(loader)).tolist()
        return (
            cast(float, averages[0]),
            cast(float, averages[1]),
            cast(float, averages[2]),
        )

    def _step(
        self, batch: dict[str, Tensor], training: bool
    ) -> tuple[float, float, float]:
        """Execute one forward pass and optional optimization step."""
        images, masks = batch["image"].to(self.device), batch["mask"].to(self.device)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, enabled=self.scaler.is_enabled()
        ):
            logits = self.model(images)
            loss = combined_loss(logits, masks)
        if training:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()  # type: ignore[no-untyped-call]
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return (
            loss.item(),
            dice_score(logits, masks).mean().item(),
            iou_score(logits, masks).mean().item(),
        )

    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, dice: float, iou: float
    ) -> None:
        """Log the required training metrics through the supplied MLflow client."""
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dsc": dice,
            "val_iou": iou,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, key, value, step=epoch)
