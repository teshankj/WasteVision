"""Train the WasteVision MobileNetV3-Small baseline."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wastevision.data.dataset import (
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


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def create_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    data_config = config["data"]
    training_config = config["training"]

    train_dataset = WasteDataset(
        data_config["train_manifest"],
        transform=get_train_transforms(),
    )

    val_dataset = WasteDataset(
        data_config["val_manifest"],
        transform=get_eval_transforms(),
    )

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
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

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    validation_loss: float,
    config: dict,
) -> None:
    """Save the best model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_loss": validation_loss,
        "config": config,
    }

    torch.save(checkpoint, path)


def train(
    config: dict,
    device: torch.device,
) -> None:
    """Train the baseline model."""
    seed = int(config["seed"])
    model_config = config["model"]
    training_config = config["training"]

    set_seed(seed)

    train_loader, val_loader = create_dataloaders(config)

    model = WasteClassifier(
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
        freeze_backbone=bool(model_config["freeze_backbone"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    epochs = int(training_config["epochs"])

    checkpoint_path = Path(
        training_config["checkpoint_path"]
    )

    best_validation_loss = float("inf")

    LOGGER.info("Device: %s", device)
    LOGGER.info("Training samples: %d", len(train_loader.dataset))
    LOGGER.info("Validation samples: %d", len(val_loader.dataset))
    LOGGER.info("Batch size: %d", training_config["batch_size"])
    LOGGER.info("Epochs: %d", epochs)
    LOGGER.info("Learning rate: %s", training_config["learning_rate"])

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        LOGGER.info(
            "Epoch %d/%d | "
            "Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f",
            epoch,
            epochs,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model=model,
                path=checkpoint_path,
                epoch=epoch,
                validation_loss=validation_loss,
                config=config,
            )

            LOGGER.info(
                "Saved best checkpoint: %s",
                checkpoint_path,
            )

    LOGGER.info(
        "Training complete. Best validation loss: %.4f",
        best_validation_loss,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the WasteVision baseline model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to the training configuration.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda or cpu.",
    )

    return parser.parse_args()


def main() -> None:
    """Run baseline training."""
    configure_logging()

    args = parse_args()
    config = load_config(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    train(
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()