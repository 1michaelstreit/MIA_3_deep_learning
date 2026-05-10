"""Training script for the BreastMNIST deep learning assignment.

Train your model and save it to models/submission_cnn.pt:

    python train.py       # or: make train

The script reads hyperparameters from config.yaml, imports your model
architecture from model.py, trains on BreastMNIST, and saves the trained
weights to models/submission_cnn.pt.

After training, commit the saved model and push:

    git add models/submission_cnn.pt
    git commit -m "Add trained model"
    git push

Do NOT modify this file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: str = "config.yaml") -> dict:
    """Load hyperparameters from config.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_medmnist_split(
    dataset_name: str, split: str, data_root: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one split of a MedMNIST dataset; returns (images, labels)."""
    import medmnist
    from medmnist import INFO

    info = INFO[dataset_name]
    class_name = str(info["python_class"])
    dataset_cls = getattr(medmnist, class_name)

    Path(data_root).mkdir(parents=True, exist_ok=True)
    ds = dataset_cls(split=split, download=True, root=data_root)

    images = np.asarray(ds.imgs, dtype=np.float32)
    labels = np.asarray(ds.labels, dtype=np.int64).reshape(-1)

    if images.ndim == 3:  # (N, H, W) → (N, H, W, 1)
        images = images[:, :, :, None]

    images = images / 255.0
    return images, labels


def load_data(
    dataset_name: str = "breastmnist", data_root: str = "./data"
) -> Dict[str, np.ndarray]:
    """Return a dict with train/val/test images and labels."""
    train_images, train_labels = _load_medmnist_split(dataset_name, "train", data_root)
    val_images, val_labels = _load_medmnist_split(dataset_name, "val", data_root)
    test_images, test_labels = _load_medmnist_split(dataset_name, "test", data_root)
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert (N, H, W, 1) float32 numpy array → (N, 1, H, W) tensor."""
    gray = images[..., 0] if images.shape[-1] == 1 else images.mean(axis=-1)
    return torch.from_numpy(gray[:, None, :, :].astype(np.float32))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return float((preds == labels).float().mean().item())


def train_model(
    model: nn.Module,
    config: dict,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    verbose: bool = True,
) -> nn.Module:
    """Train *model* in-place and return it."""
    t_cfg = config["training"]
    batch_size: int = int(t_cfg["batch_size"])
    lr: float = float(t_cfg["learning_rate"])
    num_epochs: int = int(t_cfg["num_epochs"])
    seed: int = int(t_cfg["random_seed"])

    set_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)

    x_train = _to_tensor(train_images)
    y_train = torch.from_numpy(train_labels.astype(np.float32))
    x_val = _to_tensor(val_images)
    y_val = torch.from_numpy(val_labels.astype(np.float32))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Balance the loss for the class imbalance present in BreastMNIST
    # (class 1 ≈ 73 %, class 0 ≈ 27 %).  pos_weight < 1 down-weights the
    # majority class so both classes contribute equally to the gradient.
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = -1.0
    best_state: dict | None = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_acc += _binary_accuracy(logits.detach(), yb)
            n_batches += 1

        model.eval()
        val_loss, val_acc, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(1)
                val_loss += float(criterion(logits, yb).item())
                val_acc += _binary_accuracy(logits, yb)
                v_batches += 1

        epoch_val_acc = val_acc / max(v_batches, 1)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"Epoch {epoch:02d}/{num_epochs}  "
                f"train_loss={train_loss / max(n_batches, 1):.4f}  "
                f"train_acc={train_acc / max(n_batches, 1):.4f}  "
                f"val_loss={val_loss / max(v_batches, 1):.4f}  "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Restore the weights from the epoch with the best validation accuracy.
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module, test_images: np.ndarray, test_labels: np.ndarray
) -> float:
    """Return test-set accuracy for *model*."""
    model.eval()
    x = _to_tensor(test_images)
    with torch.no_grad():
        logits = model(x).squeeze(1)
    preds = (torch.sigmoid(logits) >= 0.5).numpy().astype(np.int64)
    return float(np.mean(preds == test_labels))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def load_baseline_accuracy(metrics_path: str = "models/baseline_metrics.json") -> float:
    """Return the baseline CNN test-set accuracy stored in the repo."""
    with open(metrics_path, "r") as f:
        return float(json.load(f)["baseline_cnn_accuracy"])


def compute_score(student_accuracy: float, baseline_accuracy: float) -> float:
    """Map student accuracy to a score in [5, 15].

    - student_accuracy <= baseline_accuracy  →  5.0
    - student_accuracy == 1.0                →  15.0
    - linear interpolation in between
    """
    if student_accuracy <= baseline_accuracy:
        return 5.0
    improvement = (student_accuracy - baseline_accuracy) / max(
        1.0 - baseline_accuracy, 1e-9
    )
    return float(min(5.0 + 10.0 * improvement, 15.0))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("BreastMNIST — Training")
    print("=" * 60)

    config = load_config()
    t_cfg = config["training"]
    d_cfg = config["data"]

    print(f"\nConfig: {t_cfg}")
    print(f"Loading data from {d_cfg['data_root']} ...")
    data = load_data(dataset_name=d_cfg["dataset_name"], data_root=d_cfg["data_root"])
    print(
        f"  train={data['train_images'].shape[0]}  "
        f"val={data['val_images'].shape[0]}  "
        f"test={data['test_images'].shape[0]}"
    )

    from model import BreastLesionCNN

    set_seed(int(t_cfg["random_seed"]))
    model = BreastLesionCNN(num_classes=1)

    print("\nTraining ...")
    model = train_model(
        model,
        config,
        data["train_images"],
        data["train_labels"],
        data["val_images"],
        data["val_labels"],
        verbose=True,
    )

    accuracy = evaluate_model(model, data["test_images"], data["test_labels"])
    baseline_accuracy = load_baseline_accuracy()
    score = compute_score(accuracy, baseline_accuracy)

    # Save trained model
    Path("models").mkdir(exist_ok=True)
    weights_path = Path("models/submission_cnn.pt")
    torch.save(model.state_dict(), weights_path)
    with open("models/submission_metrics.json", "w") as f:
        json.dump({"test_accuracy": accuracy, "score": score}, f, indent=2)

    size_mb = weights_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"Test accuracy  : {accuracy:.4f}")
    print(f"Reference acc  : {baseline_accuracy:.4f}")
    print(f"Score          : {score:.1f} / 15")
    print(f"Model saved    : {weights_path}  ({size_mb:.2f} MB)")
    print("=" * 60)

    if size_mb > 10.0:
        print(
            f"\nWARNING: model file is {size_mb:.2f} MB — exceeds the 10 MB limit."
            "\nReduce the number of parameters and retrain before submitting."
        )
    else:
        print("\nNext step: commit and push your model.")
        print("  git add models/submission_cnn.pt")
        print('  git commit -m "Add trained model"')
        print("  git push")


if __name__ == "__main__":
    main()
