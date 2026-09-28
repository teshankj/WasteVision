"""Create reproducible stratified train, validation, and test splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def create_splits(
    manifest_path: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    """Create reproducible stratified train/validation/test manifests."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1.0."
        )

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be greater than zero.")

    manifest = pd.read_csv(manifest_path)

    if manifest.empty:
        raise ValueError("Manifest is empty.")

    train_df, temporary_df = train_test_split(
        manifest,
        test_size=val_ratio + test_ratio,
        stratify=manifest["class_name"],
        random_state=seed,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=relative_test_ratio,
        stratify=temporary_df["class_name"],
        random_state=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    LOGGER.info("Train samples:      %d", len(train_df))
    LOGGER.info("Validation samples: %d", len(val_df))
    LOGGER.info("Test samples:       %d", len(test_df))
    LOGGER.info("Total samples:      %d", len(train_df) + len(val_df) + len(test_df))

    _log_distribution("Train", train_df)
    _log_distribution("Validation", val_df)
    _log_distribution("Test", test_df)


def _log_distribution(name: str, dataframe: pd.DataFrame) -> None:
    """Log class distribution for a split."""
    LOGGER.info("%s distribution:", name)

    counts = dataframe["class_name"].value_counts().sort_index()

    for class_name, count in counts.items():
        LOGGER.info("  %-10s %4d", class_name, count)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create stratified dataset splits."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
        help="Path to dataset manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for split manifests.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    """Run dataset splitting."""
    configure_logging()
    args = parse_args()

    create_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()