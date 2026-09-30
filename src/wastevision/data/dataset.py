"""PyTorch dataset and image transforms for WasteVision."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES: Final[tuple[str, ...]] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

IMAGE_SIZE: Final[int] = 224

IMAGENET_MEAN: Final[tuple[float, float, float]] = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD: Final[tuple[float, float, float]] = (
    0.229,
    0.224,
    0.225,
)


def get_baseline_train_transforms() -> transforms.Compose:
    """Return the original baseline training transforms."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_augmented_train_transforms() -> transforms.Compose:
    """Return augmented training transforms for Experiment 2."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_train_transforms(
    augmentation: bool = False,
) -> transforms.Compose:
    """Return the configured training transforms."""
    if augmentation:
        return get_augmented_train_transforms()

    return get_baseline_train_transforms()


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic transforms for validation and testing."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


class WasteDataset(Dataset[tuple[Tensor, int]]):
    """Dataset backed by a WasteVision CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.data = pd.read_csv(self.manifest_path)

        required_columns = {"filepath", "class_name", "label"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )

        valid_labels = set(range(len(CLASS_NAMES)))

        if not self.data["label"].isin(valid_labels).all():
            raise ValueError("Manifest contains invalid class labels.")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load and transform one image."""
        row = self.data.iloc[index]

        image_path = Path(row["filepath"])
        label = int(row["label"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label