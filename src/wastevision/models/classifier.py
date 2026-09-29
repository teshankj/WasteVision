"""MobileNetV3-Small classifier for WasteVision."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


NUM_CLASSES = 6


class WasteClassifier(nn.Module):
    """MobileNetV3-Small transfer-learning classifier."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

        classifier_input_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Linear(
            classifier_input_features,
            num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass."""
        return self.backbone(x)