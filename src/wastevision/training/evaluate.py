"""Evaluate the trained WasteVision classifier on the test set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.tracking import MlflowClient
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


def load_config(config_path: Path) -> dict[str, Any]:
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
    config: dict[str, Any],
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

    if total_samples == 0:
        raise ValueError("Test DataLoader contains no samples.")

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
) -> tuple[dict[str, float], Path, Path]:
    """Calculate and save test metrics and evaluation artifacts."""
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

    metrics = {
        "test_loss": float(loss),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    report_dict = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    LOGGER.info("\nClassification report:\n%s", report_text)

    LOGGER.info("Test loss: %.4f", loss)
    LOGGER.info("Test accuracy: %.4f", accuracy)
    LOGGER.info("Macro precision: %.4f", macro_precision)
    LOGGER.info("Macro recall: %.4f", macro_recall)
    LOGGER.info("Macro F1: %.4f", macro_f1)
    LOGGER.info("Weighted precision: %.4f", weighted_precision)
    LOGGER.info("Weighted recall: %.4f", weighted_recall)
    LOGGER.info("Weighted F1: %.4f", weighted_f1)

    metrics_path = output_dir / "metrics.csv"

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
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

    confusion_matrix_path = output_dir / "confusion_matrix.csv"

    confusion_df.to_csv(confusion_matrix_path)

    classification_report_path = output_dir / "classification_report.csv"

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.index.name = "class"
    report_df.to_csv(classification_report_path)

    LOGGER.info(
        "Metrics saved to: %s",
        metrics_path,
    )

    LOGGER.info(
        "Confusion matrix saved to: %s",
        confusion_matrix_path,
    )

    LOGGER.info(
        "Classification report saved to: %s",
        classification_report_path,
    )

    return (
        metrics,
        metrics_path,
        confusion_matrix_path,
    )


def log_evaluation_to_mlflow(
    run_id: str,
    tracking_uri: str,
    metrics: dict[str, float],
    metrics_path: Path,
    confusion_matrix_path: Path,
    classification_report_path: Path,
) -> None:
    """Log test metrics and evaluation artifacts to an existing MLflow run."""
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(run_id)

    LOGGER.info(
        "Logging test evaluation to MLflow run: %s",
        run_id,
    )

    for key, value in metrics.items():
        metric_name = key if key.startswith("test_") else f"test_{key}"
        client.log_metric(
            run_id=run_id,
            key=metric_name,
            value=float(value),
        )

    client.log_artifact(
        run_id=run_id,
        local_path=str(metrics_path),
        artifact_path="evaluation",
    )

    client.log_artifact(
        run_id=run_id,
        local_path=str(confusion_matrix_path),
        artifact_path="evaluation",
    )

    client.log_artifact(
        run_id=run_id,
        local_path=str(classification_report_path),
        artifact_path="evaluation",
    )

    LOGGER.info(
        "MLflow evaluation logging complete. Run status: %s",
        run.info.status,
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
        default=None,
        help=(
            "Path to trained model checkpoint. "
            "Defaults to training.checkpoint_path from the config."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/mlflow_baseline"),
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda or cpu.",
    )

    parser.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help=(
            "Existing MLflow run ID to which test metrics and "
            "evaluation artifacts should be logged."
        ),
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
    mlflow_config = config.get("mlflow", {})

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        checkpoint_path = Path(
            training_config["checkpoint_path"]
        )

    test_dataset = WasteDataset(
        data_config["test_manifest"],
        transform=get_eval_transforms(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    LOGGER.info("Device: %s", device)
    LOGGER.info("Test samples: %d", len(test_dataset))
    LOGGER.info("Checkpoint: %s", checkpoint_path)

    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )

    labels, predictions, loss = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics, metrics_path, confusion_matrix_path = save_metrics(
        labels=labels,
        predictions=predictions,
        loss=loss,
        output_dir=args.output_dir,
    )

    if args.mlflow_run_id:
        tracking_uri = mlflow_config.get(
            "tracking_uri",
            "sqlite:///mlflow.db",
        )

        classification_report_path = (
            args.output_dir / "classification_report.csv"
        )

        log_evaluation_to_mlflow(
            run_id=args.mlflow_run_id,
            tracking_uri=tracking_uri,
            metrics=metrics,
            metrics_path=metrics_path,
            confusion_matrix_path=confusion_matrix_path,
            classification_report_path=classification_report_path,
        )
    else:
        LOGGER.info(
            "No MLflow run ID supplied. "
            "Evaluation was saved locally only."
        )


if __name__ == "__main__":
    main()