"""Evaluate the trained WasteVision classifier on the test set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from wastevision.data.dataset import (
    CLASS_NAMES,
    WasteDataset,
    get_eval_transforms,
)
from wastevision.models.classifier import WasteClassifier


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def load_checkpoint(
    checkpoint_path: Path,
    config: dict,
    device: torch.device,
) -> WasteClassifier:
    """Load the trained classifier from a checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model_config = config["model"]

    model = WasteClassifier(
        num_classes=int(model_config["num_classes"]),
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    LOGGER.info(
        "Loaded checkpoint from epoch %d",
        checkpoint["epoch"],
    )

    LOGGER.info(
        "Checkpoint validation loss: %.4f",
        checkpoint["validation_loss"],
    )

    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference on the test set."""
    criterion = nn.CrossEntropyLoss()

    all_predictions: list[int] = []
    all_labels: list[int] = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

    average_loss = total_loss / total_samples

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        average_loss,
    )


def save_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    loss: float,
    output_dir: Path,
) -> None:
    """Save evaluation metrics and confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(labels, predictions)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    LOGGER.info("\nClassification report:\n%s", report)

    LOGGER.info("Test loss: %.4f", loss)
    LOGGER.info("Test accuracy: %.4f", accuracy)
    LOGGER.info("Macro precision: %.4f", macro_precision)
    LOGGER.info("Macro recall: %.4f", macro_recall)
    LOGGER.info("Macro F1: %.4f", macro_f1)
    LOGGER.info("Weighted precision: %.4f", weighted_precision)
    LOGGER.info("Weighted recall: %.4f", weighted_recall)
    LOGGER.info("Weighted F1: %.4f", weighted_f1)

    metrics = {
        "test_loss": loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    pd.DataFrame([metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    confusion_df.index.name = "actual"

    confusion_df.to_csv(
        output_dir / "confusion_matrix.csv",
    )

    LOGGER.info(
        "Metrics saved to: %s",
        output_dir / "metrics.csv",
    )

    LOGGER.info(
        "Confusion matrix saved to: %s",
        output_dir / "confusion_matrix.csv",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the WasteVision model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/baseline/best_model.pt"),
        help="Path to trained model checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/baseline"),
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda or cpu.",
    )

    return parser.parse_args()


def main() -> None:
    """Run model evaluation."""
    configure_logging()

    args = parse_args()
    config = load_config(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    data_config = config["data"]
    training_config = config["training"]

    test_dataset = WasteDataset(
        data_config["test_manifest"],
        transform=get_eval_transforms(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    LOGGER.info("Device: %s", device)
    LOGGER.info("Test samples: %d", len(test_dataset))

    model = load_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
    )

    labels, predictions, loss = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    save_metrics(
        labels=labels,
        predictions=predictions,
        loss=loss,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()