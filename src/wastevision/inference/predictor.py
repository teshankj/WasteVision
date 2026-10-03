"""WasteVision model inference utilities."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import torch
from PIL import Image
from torch import Tensor

from wastevision.data.dataset import CLASS_NAMES, get_eval_transforms
from wastevision.models.classifier import WasteClassifier


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: Final[Path] = Path(
    "artifacts/mlflow_baseline/best_model.pt"
)

MODEL_NAME: Final[str] = "wastevision"
MODEL_VERSION: Final[str] = "1"


class WastePredictor:
    """Load the trained WasteVision model and perform inference."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        self.device = torch.device("cpu")

        self.transform = get_eval_transforms()

        self.model = WasteClassifier(
            num_classes=len(CLASS_NAMES),
            pretrained=False,
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dict'."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        LOGGER.info(
            "Loaded model '%s' version '%s' from %s",
            MODEL_NAME,
            MODEL_VERSION,
            self.model_path,
        )

    def predict(
        self,
        image: Image.Image,
    ) -> tuple[str, float, float]:
        """Predict the waste class for one image.

        Returns:
            A tuple containing predicted class, confidence, and
            inference time in milliseconds.
        """
        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        with torch.inference_mode():
            logits: Tensor = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        inference_time_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

        predicted_class = CLASS_NAMES[predicted_index.item()]

        return (
            predicted_class,
            float(confidence.item()),
            inference_time_ms,
        )