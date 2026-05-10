"""Evaluate a trained model on the BreastMNIST test set.

Usage:
    python evaluate.py             # evaluate your trained model
    python evaluate.py --baseline  # evaluate the pre-trained reference model

This script loads a saved .pt file and runs inference — it does NOT retrain.
Make sure you have run 'make train' before running 'make evaluate'.

Do NOT modify this file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Limits enforced by the grader
# ---------------------------------------------------------------------------

MODEL_SIZE_LIMIT_MB: float = 10.0
INFERENCE_TIME_LIMIT_S: float = 30.0

# ---------------------------------------------------------------------------
# Reference architecture (fixed — do not modify)
#
# This class mirrors the architecture used to produce models/baseline_cnn.pt.
# It is kept here so the baseline can always be loaded, independent of any
# changes you make to model.py.
# ---------------------------------------------------------------------------


class _BaselineCNN(nn.Module):
    """Pre-trained reference CNN.  Do not edit.

    This architecture must stay in sync with the default BreastLesionCNN
    defined in model.py — it is used to load models/baseline_cnn.pt.
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Single conv block — 28×28 → 14×14
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Core helpers (imported by tests and train.py)
# ---------------------------------------------------------------------------


def get_model_size_mb(path: str | Path) -> float:
    """Return the size of *path* in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)


def load_model(model_cls: type, weights_path: str | Path) -> nn.Module:
    """Instantiate *model_cls* and load weights from *weights_path*.

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {weights_path}\n"
            "Run 'make train' to generate it, then commit the file before pushing."
        )
    model = model_cls(num_classes=1)
    model.load_state_dict(
        torch.load(str(weights_path), map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def run_inference(
    model: nn.Module, test_images: np.ndarray, test_labels: np.ndarray
) -> tuple[float, float]:
    """Return (accuracy, elapsed_seconds) for inference on the test set."""
    from train import _to_tensor

    model.eval()
    x = _to_tensor(test_images)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x).squeeze(1)
    elapsed = time.perf_counter() - t0
    preds = (torch.sigmoid(logits) >= 0.5).numpy().astype(np.int64)
    accuracy = float(np.mean(preds == test_labels))
    return accuracy, elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a BreastMNIST model on the test set"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the pre-trained reference model instead of your own",
    )
    args = parser.parse_args()

    from train import compute_score, load_baseline_accuracy, load_config, load_data

    config = load_config(str(ROOT / "config.yaml"))
    d_cfg = config["data"]
    print(f"Loading data from {d_cfg['data_root']} ...")
    data = load_data(dataset_name=d_cfg["dataset_name"], data_root=d_cfg["data_root"])
    baseline_accuracy = load_baseline_accuracy(
        str(ROOT / "models" / "baseline_metrics.json")
    )

    if args.baseline:
        label = "Reference"
        weights_path = ROOT / "models" / "baseline_cnn.pt"
        model = load_model(_BaselineCNN, weights_path)
    else:
        label = "Your"
        weights_path = ROOT / "models" / "submission_cnn.pt"
        from model import BreastLesionCNN

        model = load_model(BreastLesionCNN, weights_path)

    size_mb = get_model_size_mb(weights_path)
    accuracy, elapsed = run_inference(model, data["test_images"], data["test_labels"])

    print("\n" + "=" * 60)
    print(f"{label} model — evaluation results")
    print("=" * 60)
    print(
        f"Model file     : {weights_path.name}  ({size_mb:.2f} MB / {MODEL_SIZE_LIMIT_MB:.0f} MB limit)"
    )
    print(
        f"Inference time : {elapsed:.3f} s  ({len(data['test_labels'])} images, limit {INFERENCE_TIME_LIMIT_S:.0f} s)"
    )
    print(f"Test accuracy  : {accuracy:.4f}")
    if not args.baseline:
        score = compute_score(accuracy, baseline_accuracy)
        print(f"Reference acc  : {baseline_accuracy:.4f}")
        print(f"Score          : {score:.1f} / 15")
    print("=" * 60)

    if size_mb > MODEL_SIZE_LIMIT_MB:
        print(
            f"\nWARNING: model file is {size_mb:.2f} MB — exceeds the {MODEL_SIZE_LIMIT_MB:.0f} MB limit."
            "\nReduce the number of parameters and retrain before submitting."
        )
    if elapsed > INFERENCE_TIME_LIMIT_S:
        print(
            f"\nWARNING: inference took {elapsed:.1f} s — exceeds the {INFERENCE_TIME_LIMIT_S:.0f} s limit."
        )


if __name__ == "__main__":
    main()
