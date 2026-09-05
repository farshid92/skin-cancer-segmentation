"""Tests for canonical raw-dataset preparation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.preparation import create_splits, prepare_dataset


def test_prepare_dataset_writes_canonical_pairs_and_splits(tmp_path: Path) -> None:
    """Preparation writes canonical names, binary masks, and all split partitions."""
    image_dir, mask_dir = _write_raw_pairs(tmp_path, count=10)

    splits = prepare_dataset(
        image_dir, mask_dir, tmp_path / "processed", "isic2018", 42
    )

    assert {name: len(records) for name, records in splits.items()} == {
        "train": 7,
        "val": 1,
        "test": 2,
    }
    record = splits["train"][0]
    assert record["image"].startswith("images/isic2018_ISIC_")
    mask = cv2.imread(
        str(tmp_path / "processed" / record["mask"]), cv2.IMREAD_GRAYSCALE
    )
    assert set(np.unique(mask).tolist()) <= {0, 255}


def test_prepare_dataset_rejects_misaligned_pairs(tmp_path: Path) -> None:
    """Preparation refuses images and masks with incompatible dimensions."""
    image_dir, mask_dir = _write_raw_pairs(tmp_path, count=3)
    cv2.imwrite(
        str(mask_dir / "ISIC_0000000_segmentation.png"), np.zeros((5, 5), np.uint8)
    )

    with pytest.raises(ValueError, match="dimensions differ"):
        prepare_dataset(image_dir, mask_dir, tmp_path / "processed", "isic2018", 42)


def test_create_splits_is_reproducible() -> None:
    """A seed creates stable 70/10/20 sample assignments."""
    records = [{"image": str(index), "mask": str(index)} for index in range(10)]

    assert create_splits(records, 42) == create_splits(records, 42)


def _write_raw_pairs(root: Path, count: int) -> tuple[Path, Path]:
    """Create small ISIC-style raw image and mask files."""
    image_dir = root / "raw_images"
    mask_dir = root / "raw_masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    for index in range(count):
        identifier = f"ISIC_{index:07d}"
        image = np.full((8, 10, 3), index, dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:7] = 255
        cv2.imwrite(str(image_dir / f"{identifier}.jpg"), image)
        cv2.imwrite(str(mask_dir / f"{identifier}_segmentation.png"), mask)
    return image_dir, mask_dir
