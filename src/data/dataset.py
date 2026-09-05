"""Split-driven dermoscopic lesion dataset and data-loader construction."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import prepare_image, prepare_mask, remove_hair


class LesionDataset(Dataset[dict[str, torch.Tensor]]):
    """Load aligned lesion images and binary masks from split records."""

    def __init__(
        self,
        root: Path,
        records: Sequence[dict[str, str]],
        image_size: int,
        remove_hair_enabled: bool = True,
    ) -> None:
        self.root = root
        self.records = list(records)
        self.image_size = image_size
        self.remove_hair_enabled = remove_hair_enabled

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one image and mask pair as documented float32 tensors."""
        record = self.records[index]
        image = self._read_image(record["image"])
        if self.remove_hair_enabled:
            image = remove_hair(image)
        mask = self._read_mask(record["mask"])
        return {
            "image": prepare_image(image, self.image_size),
            "mask": prepare_mask(mask, self.image_size),
        }

    def _read_image(self, relative_path: str) -> np.ndarray:
        """Read an RGB image relative to the dataset root."""
        image_path = self.root / relative_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, relative_path: str) -> np.ndarray:
        """Read a grayscale segmentation mask relative to the dataset root."""
        mask_path = self.root / relative_path
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found or unreadable: {mask_path}")
        return mask


def load_splits(split_path: Path) -> dict[str, list[dict[str, str]]]:
    """Load and validate the persisted train, validation, and test splits."""
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open(encoding="utf-8") as stream:
        raw_splits = cast(dict[str, Any], json.load(stream))
    required_splits = {"train", "val", "test"}
    if set(raw_splits) != required_splits:
        raise ValueError("Split file must contain exactly: train, val, test.")
    return {name: _validate_records(name, raw_splits[name]) for name in required_splits}


def build_loaders(config: DictConfig) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    """Build deterministic train, validation, and test data loaders from config."""
    data_config = config.data
    root = Path(cast(str, data_config.root))
    splits = load_splits(Path(cast(str, data_config.split_path)))
    generator = torch.Generator().manual_seed(cast(int, config.seed))
    return {
        name: DataLoader(
            LesionDataset(
                root=root,
                records=records,
                image_size=cast(int, data_config.image_size),
                remove_hair_enabled=cast(bool, data_config.remove_hair),
            ),
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            batch_size=cast(int, data_config.batch_size),
            num_workers=cast(int, data_config.num_workers),
            pin_memory=torch.cuda.is_available(),
        )
        for name, records in splits.items()
    }


def _validate_records(name: str, records: Any) -> list[dict[str, str]]:
    """Validate that split records contain string image and mask paths."""
    if not isinstance(records, list):
        raise TypeError(f"Split '{name}' must be a list of records.")
    validated: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"image", "mask"}:
            raise ValueError(f"Split '{name}' records require image and mask paths.")
        image_path, mask_path = record["image"], record["mask"]
        if not isinstance(image_path, str) or not isinstance(mask_path, str):
            raise TypeError(f"Split '{name}' paths must be strings.")
        validated.append({"image": image_path, "mask": mask_path})
    return validated
