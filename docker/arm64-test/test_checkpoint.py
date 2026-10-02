from pathlib import Path
import platform
import time

import torch

from wastevision.models.classifier import WasteClassifier


CHECKPOINT_PATH = Path(
    "/workspace/artifacts/mlflow_baseline/best_model.pt"
)

CLASS_NAMES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)


def main() -> None:
    print("=" * 60)
    print("WasteVision ARM64 Checkpoint Compatibility Test")
    print("=" * 60)

    print(f"Architecture: {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Validation loss: "
        f"{checkpoint['validation_loss']:.6f}"
    )

    model = WasteClassifier(
        num_classes=6,
        pretrained=False,
        freeze_backbone=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        "State dict: compatible"
    )

    dummy_input = torch.randn(1, 3, 224, 224)

    # Warm-up
    with torch.inference_mode():
        _ = model(dummy_input)

    start = time.perf_counter()

    with torch.inference_mode():
        output = model(dummy_input)

    elapsed_ms = (time.perf_counter() - start) * 1000

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_index = torch.max(
        probabilities, dim=1
    )

    predicted_class = CLASS_NAMES[predicted_index.item()]

    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    print(f"Inference time: {elapsed_ms:.2f} ms")

    assert output.shape == (1, 6)
    assert torch.isfinite(output).all()
    assert torch.isfinite(probabilities).all()

    print("=" * 60)
    print("ARM64 CHECKPOINT COMPATIBILITY: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()