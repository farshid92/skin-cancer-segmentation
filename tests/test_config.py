"""Tests for the project configuration contract."""

from pathlib import Path

import pytest

from src.utils.config import load_config, to_container

ROOT = Path(__file__).parents[1]


def test_base_config_loads_and_supports_overrides() -> None:
    """Base settings load and CLI-style overrides are resolved."""
    config = load_config([ROOT / "configs" / "base.yaml"], ["data.batch_size=2"])

    assert config.seed == 42
    assert config.data.batch_size == 2
    assert to_container(config)["model"]["name"] == "unetpp"


def test_missing_config_is_reported() -> None:
    """Missing configuration paths fail with an actionable exception."""
    with pytest.raises(FileNotFoundError):
        load_config([ROOT / "configs" / "missing.yaml"])
