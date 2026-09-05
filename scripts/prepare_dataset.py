"""Prepare one raw dermoscopic dataset into canonical training files."""

import argparse
from pathlib import Path

from src.data.preparation import prepare_dataset
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse raw input locations and preparation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Prepare a dataset and log its persisted split counts."""
    args = parse_args()
    set_seed(args.seed)
    splits = prepare_dataset(
        args.images, args.masks, args.output, args.dataset, args.seed
    )
    counts = ", ".join(f"{name}={len(records)}" for name, records in splits.items())
    get_logger(__name__).info("Prepared %s: %s", args.dataset, counts)


if __name__ == "__main__":
    main()
