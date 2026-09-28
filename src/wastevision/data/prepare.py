"""Validate the WasteVision dataset and create a dataset manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

LOGGER = logging.getLogger(__name__)

EXPECTED_CLASSES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def validate_image(path: Path) -> bool:
    """Return True when an image can be opened and verified."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception as exc:
        LOGGER.warning("Invalid image: %s (%s)", path, exc)
        return False


def discover_images(raw_dir: Path) -> list[tuple[Path, str]]:
    """Discover valid image files and associate them with their class."""
    discovered: list[tuple[Path, str]] = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue

        if any(part.startswith(".") for part in path.parts):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_parts = path.relative_to(raw_dir).parts

        if len(relative_parts) < 2:
            LOGGER.warning("Skipping file without class directory: %s", path)
            continue

        class_name = relative_parts[0]

        if class_name not in EXPECTED_CLASSES:
            LOGGER.warning("Skipping unexpected class: %s", path)
            continue

        if validate_image(path):
            discovered.append((path, class_name))

    return discovered


def create_manifest(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    """Validate the dataset and write its manifest."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    discovered = discover_images(raw_dir)

    if not discovered:
        raise RuntimeError("No valid images were found.")

    discovered_classes = {class_name for _, class_name in discovered}
    missing_classes = set(EXPECTED_CLASSES) - discovered_classes

    if missing_classes:
        raise RuntimeError(
            f"Missing expected classes: {sorted(missing_classes)}"
        )

    records = [
        {
            "filepath": str(path),
            "class_name": class_name,
            "label": EXPECTED_CLASSES.index(class_name),
        }
        for path, class_name in discovered
    ]

    manifest = pd.DataFrame(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    return manifest


def log_dataset_summary(manifest: pd.DataFrame) -> None:
    """Log dataset statistics."""
    LOGGER.info("Total valid images: %d", len(manifest))

    counts = manifest["class_name"].value_counts().reindex(
        EXPECTED_CLASSES,
        fill_value=0,
    )

    LOGGER.info("Class distribution:")
    for class_name, count in counts.items():
        percentage = count / len(manifest) * 100
        LOGGER.info(
            "  %-10s %4d images (%.2f%%)",
            class_name,
            count,
            percentage,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate WasteVision dataset and create a manifest."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/dataset-original"),
        help="Path to the immutable raw dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifest.csv"),
        help="Output manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset preparation."""
    configure_logging()
    args = parse_args()

    manifest = create_manifest(
        raw_dir=args.raw_dir,
        output_path=args.output,
    )

    log_dataset_summary(manifest)

    LOGGER.info("Manifest written to: %s", args.output)


if __name__ == "__main__":
    main()