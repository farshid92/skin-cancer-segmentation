"""Configuration loading and validation helpers."""

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def load_config(
    paths: list[str | Path], overrides: list[str] | None = None
) -> DictConfig:
    """Load YAML files in order and apply dot-list overrides."""
    if not paths:
        raise ValueError("At least one configuration path is required.")

    configs: list[DictConfig] = []
    for path in paths:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        configs.append(cast(DictConfig, OmegaConf.load(config_path)))

    merged = OmegaConf.merge(*configs)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_readonly(merged, True)
    return cast(DictConfig, merged)


def to_container(config: DictConfig) -> dict[str, Any]:
    """Convert a loaded configuration into a regular Python dictionary."""
    result = OmegaConf.to_container(config, resolve=True)
    if not isinstance(result, dict):
        raise TypeError("The root configuration must be a mapping.")
    return cast(dict[str, Any], result)
