"""Canonical dataset preparation and persisted split generation."""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def prepare_dataset(
    image_dir: Path,
    mask_dir: Path,
    output_root: Path,
    dataset_name: str,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Create validated canonical pairs and deterministic train, val, test splits."""
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Image and mask directories must both exist.")
    split_path = output_root / "splits.json"
    if split_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split file: {split_path}"
        )
    records = _copy_canonical_pairs(image_dir, mask_dir, output_root, dataset_name)
    splits = create_splits(records, seed)
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def create_splits(
    records: Sequence[dict[str, str]], seed: int
) -> dict[str, list[dict[str, str]]]:
    """Create a seeded 70/10/20 partition from validated sample records."""
    if len(records) < 3:
        raise ValueError(
            "At least three image-mask pairs are required to create splits."
        )
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.7)
    validation_end = train_end + round(len(shuffled) * 0.1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def _copy_canonical_pairs(
    image_dir: Path, mask_dir: Path, output_root: Path, dataset_name: str
) -> list[dict[str, str]]:
    """Validate input pairs and write canonical RGB image and binary mask PNG files."""
    output_images = output_root / "images"
    output_masks = output_root / "masks"
    output_images.mkdir(parents=True, exist_ok=False)
    output_masks.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, str]] = []
    for image_path in sorted(_image_paths(image_dir)):
        mask_path = _find_mask(mask_dir, image_path.stem)
        image, mask = _read_aligned_pair(image_path, mask_path)
        sample_name = f"{dataset_name}_{image_path.stem}"
        image_target = output_images / f"{sample_name}.png"
        mask_target = output_masks / f"{sample_name}.png"
        cv2.imwrite(str(image_target), image)
        cv2.imwrite(str(mask_target), (mask > 0).astype("uint8") * 255)
        records.append(
            {
                "image": f"images/{image_target.name}",
                "mask": f"masks/{mask_target.name}",
            }
        )
    if not records:
        raise ValueError(f"No supported images found in: {image_dir}")
    return records


def _image_paths(directory: Path) -> list[Path]:
    """Return supported image files from an input directory."""
    suffixes = {".jpg", ".jpeg", ".png"}
    return [path for path in directory.iterdir() if path.suffix.lower() in suffixes]


def _find_mask(mask_dir: Path, image_stem: str) -> Path:
    """Find the matching ISIC-style segmentation mask for an image identifier."""
    candidates = [
        mask_dir / f"{image_stem}_segmentation.png",
        mask_dir / f"{image_stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask found for image identifier: {image_stem}")


def _read_aligned_pair(
    image_path: Path, mask_path: Path
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Read a pair and verify their native spatial dimensions match."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Unreadable image-mask pair: {image_path.name}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image and mask dimensions differ for: {image_path.name}")
    return (
        np.asarray(image, dtype=np.uint8),
        np.asarray(mask, dtype=np.uint8),
    )
