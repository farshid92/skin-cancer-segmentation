"""Train a configured lesion segmentation model on persisted data splits."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from torch.optim import AdamW

from src.data.dataset import build_loaders
from src.models.factory import build_model
from src.training.scheduler import build_warmup_cosine_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config, to_container
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse configuration paths and optional dot-list overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def create_run(client: MlflowClient, experiment_name: str, run_name: str) -> str:
    """Create or reuse the configured MLflow experiment and return a run ID."""
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(experiment_name)
    )
    return client.create_run(experiment_id, run_name=run_name).info.run_id


def build_run_name(config: dict[str, Any]) -> str:
    """Build the documented readable and content-addressed MLflow run name."""
    model = cast(dict[str, Any], config["model"])
    config_hash = hashlib.sha1(str(sorted(config.items())).encode()).hexdigest()[:8]
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{model['name']}_{model['encoder']}_{date}_{config_hash}"


def main() -> None:
    """Run one reproducible MLflow-tracked segmentation training job."""
    args = parse_args()
    config = load_config([Path(path) for path in args.config], args.override)
    values = to_container(config)
    set_seed(cast(int, values["seed"]))
    mlflow.set_tracking_uri(cast(str, values["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    run_id = create_run(
        client, cast(str, values["project"]["experiment_name"]), build_run_name(values)
    )
    _log_run_inputs(client, run_id, values, config)
    try:
        _train(config, client, run_id)
    except Exception:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise
    client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))


def _train(config: Any, client: MlflowClient, run_id: str) -> None:
    """Construct data and optimization components, then execute training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config.training
    model = build_model(config.model)
    optimizer = AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, training.epochs, training.warmup_epochs
    )
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        client,
        run_id,
        Path(training.checkpoint_path),
        device,
        training.early_stopping_patience,
        training.amp,
        training.gradient_clip_norm,
    )
    loaders = build_loaders(config)
    best_dice = trainer.fit(
        loaders["train"], loaders["val"], training.epochs
    )
    get_logger(__name__).info(
        "Training complete on %s; best validation Dice: %.4f", device, best_dice
    )


def _log_run_inputs(
    client: MlflowClient, run_id: str, values: dict[str, Any], config: Any
) -> None:
    """Log configuration parameters and the resolved configuration artifact."""
    flattened = _flatten(values)
    client.log_batch(
        run_id,
        params=[mlflow.entities.Param(key, value) for key, value in flattened.items()],
    )
    artifact_path = Path("experiments") / "configs" / f"{run_id}.yaml"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, artifact_path)
    client.log_artifact(run_id, str(artifact_path))


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested configuration mapping into MLflow-compatible parameters."""
    flattened: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(cast(dict[str, Any], value), name))
        else:
            flattened[name] = str(value)
    return flattened


if __name__ == "__main__":
    main()
