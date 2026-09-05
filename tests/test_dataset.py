"""Tests for split-driven lesion datasets and loaders."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.dataset import LesionDataset, build_loaders, load_splits


def test_dataset_returns_documented_image_and_mask_contract(tmp_path: Path) -> None:
    """Dataset converts aligned files into float32 CHW tensors and binary masks."""
    _write_fixture_pair(tmp_path)
    dataset = LesionDataset(
        root=tmp_path,
        records=[{"image": "images/sample.png", "mask": "masks/sample.png"}],
        image_size=8,
        remove_hair_enabled=False,
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (1, 8, 8)
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}


def test_build_loaders_uses_persisted_splits(tmp_path: Path) -> None:
    """Loader construction uses saved split records without recomputing them."""
    _write_fixture_pair(tmp_path)
    splits = {
        name: [{"image": "images/sample.png", "mask": "masks/sample.png"}]
        for name in ("train", "val", "test")
    }
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits), encoding="utf-8")
    config = OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "root": str(tmp_path),
                "split_path": str(split_path),
                "image_size": 8,
                "batch_size": 1,
                "num_workers": 0,
                "remove_hair": False,
            },
        }
    )

    loaders = build_loaders(config)
    batch = next(iter(loaders["train"]))

    assert set(loaders) == {"train", "val", "test"}
    assert batch["image"].shape == (1, 3, 8, 8)
    assert batch["mask"].shape == (1, 1, 8, 8)


def test_load_splits_rejects_missing_partition(tmp_path: Path) -> None:
    """Incomplete persisted splits fail with a clear exception."""
    split_path = tmp_path / "splits.json"
    split_path.write_text('{"train": [], "val": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="train, val, test"):
        load_splits(split_path)


def _write_fixture_pair(root: Path) -> None:
    """Create one aligned RGB image and binary mask fixture."""
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.full((10, 12, 3), 127, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:7, 3:9] = 255
    cv2.imwrite(str(image_dir / "sample.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_dir / "sample.png"), mask)
