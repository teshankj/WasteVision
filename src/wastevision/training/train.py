"""Train the WasteVision MobileNetV3-Small classifier."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wastevision.data.dataset import (
    CLASS_NAMES,
    WasteDataset,
    get_eval_transforms,
    get_train_transforms,
)
from wastevision.models.classifier import WasteClassifier


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def create_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader[tuple[Tensor, int]], DataLoader[tuple[Tensor, int]], WasteDataset]:
    """Create training and validation dataloaders."""
    data_config = config["data"]
    training_config = config["training"]

    augmentation = bool(training_config.get("augmentation", False))

    train_dataset = WasteDataset(
        data_config["train_manifest"],
        transform=get_train_transforms(augmentation=augmentation),
    )

    val_dataset = WasteDataset(
        data_config["val_manifest"],
        transform=get_eval_transforms(),
    )

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset


def calculate_class_weights(
    dataset: WasteDataset,
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Calculate inverse-frequency class weights from training data."""
    class_counts = torch.bincount(
        torch.tensor(
            dataset.data["label"].to_numpy(),
            dtype=torch.long,
        ),
        minlength=num_classes,
    ).float()

    if torch.any(class_counts == 0):
        missing_classes = torch.where(class_counts == 0)[0].tolist()
        raise ValueError(
            f"Training dataset is missing classes: {missing_classes}"
        )

    weights = class_counts.sum() / (num_classes * class_counts)

    return weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None

    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("DataLoader contains no samples.")

    return total_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    validation_loss: float,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Save the best model checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "validation_loss": validation_loss,
            "config": config,
        },
        checkpoint_path,
    )

    LOGGER.info("Saved best checkpoint: %s", checkpoint_path)


def train(
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Train the WasteVision classifier."""
    seed = int(config["seed"])
    set_seed(seed)

    train_loader, val_loader, train_dataset = create_dataloaders(config)

    model_config = config["model"]
    training_config = config["training"]

    model = WasteClassifier(
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
        freeze_backbone=bool(model_config["freeze_backbone"]),
    ).to(device)

    num_classes = int(model_config["num_classes"])
    class_weighting = bool(training_config.get("class_weighting", False))

    if class_weighting:
        class_weights = calculate_class_weights(
            dataset=train_dataset,
            num_classes=num_classes,
            device=device,
        )

        class_counts = (
            train_dataset.data["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        LOGGER.info("Class counts: %s", class_counts)
        LOGGER.info(
            "Class weights: %s",
            class_weights.detach().cpu().tolist(),
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    LOGGER.info("Device: %s", device)
    LOGGER.info("Training samples: %d", len(train_loader.dataset))
    LOGGER.info("Validation samples: %d", len(val_loader.dataset))
    LOGGER.info("Batch size: %d", int(training_config["batch_size"]))
    LOGGER.info("Epochs: %d", int(training_config["epochs"]))
    LOGGER.info(
        "Learning rate: %s",
        training_config["learning_rate"],
    )
    LOGGER.info("Augmentation: %s", training_config.get("augmentation", False))
    LOGGER.info("Class weighting: %s", class_weighting)

    best_validation_loss = float("inf")

    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        LOGGER.info(
            "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f",
            epoch,
            int(training_config["epochs"]),
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model=model,
                epoch=epoch,
                validation_loss=validation_loss,
                config=config,
                checkpoint_path=Path(
                    training_config["checkpoint_path"]
                ),
            )

    LOGGER.info(
        "Training complete. Best validation loss: %.4f",
        best_validation_loss,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the WasteVision classifier."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to training configuration.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the training CLI."""
    configure_logging()

    args = parse_args()
    config = load_config(args.config)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train(config=config, device=device)


if __name__ == "__main__":
    main()