"""Tests for the WasteVision model and inference pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from wastevision.data.dataset import CLASS_NAMES
from wastevision.inference.predictor import WastePredictor
from wastevision.models.classifier import WasteClassifier


MODEL_PATH = Path("artifacts/mlflow_baseline/best_model.pt")


def test_classifier_has_six_output_classes() -> None:
    """The classifier should produce six class logits."""

    model = WasteClassifier(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    inputs = torch.randn(1, 3, 224, 224)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (1, 6)


def test_checkpoint_exists() -> None:
    """The frozen baseline checkpoint should exist."""

    assert MODEL_PATH.exists()
    assert MODEL_PATH.is_file()


def test_predictor_loads_checkpoint() -> None:
    """The production predictor should load the frozen checkpoint."""

    predictor = WastePredictor(MODEL_PATH)

    assert predictor.model is not None
    assert predictor.model.training is False


def test_predictor_returns_valid_prediction() -> None:
    """Inference should return a valid class, confidence and latency."""

    predictor = WastePredictor(MODEL_PATH)

    image = Image.new(
        "RGB",
        (224, 224),
        color=(128, 128, 128),
    )

    prediction, confidence, inference_time_ms = predictor.predict(image)

    assert prediction in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0
    assert inference_time_ms >= 0.0


@pytest.mark.parametrize("image_size", [(100, 100), (300, 200), (640, 480)])
def test_predictor_accepts_different_image_sizes(
    image_size: tuple[int, int],
) -> None:
    """The preprocessing pipeline should handle different image sizes."""

    predictor = WastePredictor(MODEL_PATH)

    image = Image.new(
        "RGB",
        image_size,
        color=(100, 150, 200),
    )

    prediction, confidence, inference_time_ms = predictor.predict(image)

    assert prediction in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0
    assert inference_time_ms >= 0.0