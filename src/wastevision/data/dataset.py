"""PyTorch datasets for the WasteVision image classification task."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Return image transformations used for training."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic image transformations for evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class WasteDataset(Dataset[tuple[Tensor, int]]):
    """Dataset backed by a WasteVision CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
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
                f"Manifest is missing columns: {sorted(missing_columns)}"
            )

        if self.data.empty:
            raise ValueError(
                f"Manifest contains no samples: {self.manifest_path}"
            )

        invalid_labels = set(self.data["label"]) - set(range(len(CLASS_NAMES)))

        if invalid_labels:
            raise ValueError(
                f"Invalid labels found: {sorted(invalid_labels)}"
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load and transform one image."""
        row = self.data.iloc[index]

        image_path = Path(row["filepath"])
        label = int(row["label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by manifest does not exist: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label