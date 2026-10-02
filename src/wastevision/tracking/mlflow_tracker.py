"""MLflow experiment tracking utilities for WasteVision."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch


LOGGER = logging.getLogger(__name__)


def configure_mlflow(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "WasteVision",
) -> None:
    """Configure the MLflow tracking backend and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    LOGGER.info("MLflow tracking URI: %s", tracking_uri)
    LOGGER.info("MLflow experiment: %s", experiment_name)


def start_run(
    run_name: str,
    parameters: dict[str, Any],
) -> mlflow.ActiveRun:
    """Start an MLflow run and log parameters."""
    run = mlflow.start_run(run_name=run_name)

    clean_parameters = {
        key: value
        for key, value in parameters.items()
        if value is not None
    }

    mlflow.log_params(clean_parameters)

    LOGGER.info("Started MLflow run: %s", run_name)

    return run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log numeric metrics to the active MLflow run."""
    if step is None:
        mlflow.log_metrics(metrics)
    else:
        mlflow.log_metrics(metrics, step=step)


def log_model(
    model: Any,
    artifact_path: str = "model",
) -> None:
    """Log a PyTorch model to the active MLflow run."""
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
    )


def log_artifact(path: str | Path) -> None:
    """Log a local file as an MLflow artifact."""
    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact does not exist: {artifact_path}"
        )

    mlflow.log_artifact(str(artifact_path))


def end_run() -> None:
    """End the active MLflow run."""
    if mlflow.active_run() is not None:
        mlflow.end_run()