[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tj1RlgK1)
# 03 — Deep Learning for Medical Image Classification

Assignment 3 for the graduate Image Analysis course.

## Overview

You will design and train a convolutional neural network to classify breast
lesions as benign or malignant using the
[BreastMNIST](https://medmnist.com/) dataset (28 × 28 grayscale images,
binary classification).

Your goal is to maximise test-set accuracy. Grading is out of **15 points**:

| Test-set accuracy | Score |
|---|---|
| ≤ reference accuracy (~0.73) | 5 / 15 |
| 1.0 (perfect) | 15 / 15 |
| anything in between | linearly interpolated |

## Files to edit (only these two)

| File | What to change |
|---|---|
| `model.py` | CNN architecture inside `BreastLesionCNN` |
| `config.yaml` | Training hyperparameters: `batch_size`, `learning_rate`, `num_epochs` |

Every other file is part of the grading scaffold — **do not modify it**.

## Constraints

| Constraint | Limit |
|---|---|
| Saved model file size | **≤ 10 MB** |
| Inference on the test set | **≤ 30 seconds** |
| `num_epochs` in config.yaml | keep ≤ 100 |

Submissions that exceed these limits will fail the corresponding grading tests.

## Workflow

### 1. Set up the environment

```bash
make setup
source .venv/bin/activate
```

Requires [uv](https://docs.astral.sh/uv/) — install it once with
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

### 2. See the reference model

```bash
make baseline
```

Runs inference with the pre-trained reference CNN (`models/baseline_cnn.pt`)
and prints its test-set accuracy. This is the score you need to beat.

### 3. Edit `model.py` and `config.yaml`

Change the architecture and/or hyperparameters. See the comments inside each
file for guidance on what you can modify.

### 4. Train your model

```bash
make train
```

Trains `BreastLesionCNN` with your `config.yaml` settings and saves the
weights to `models/submission_cnn.pt`. Training also prints per-epoch
accuracy so you can monitor progress.

### 5. Evaluate your model

```bash
make evaluate
```

Loads `models/submission_cnn.pt` — **no retraining** — runs inference on the
test set, and prints your estimated score out of 15.

### 6. Run the grading tests locally

```bash
make test
```

Runs the same tests that will run on GitHub when you push. Each of the 30
threshold tests is worth 0.5 points.

### 7. Commit and push

```bash
git add model.py config.yaml models/submission_cnn.pt
git commit -m "Improve CNN architecture and hyperparameters"
git push
```

**Important:** `models/submission_cnn.pt` must be committed. GitHub grading
loads the saved file and runs inference — it does not retrain. If the file is
missing, all grading tests will fail.

## Repository structure

```
model.py                      ← edit: CNN architecture
config.yaml                   ← edit: hyperparameters
train.py                      ← scaffold: trains and saves your model
evaluate.py                   ← scaffold: loads and evaluates your model
Makefile                      ← convenience commands
tests/
  conftest.py                 ← scaffold: pytest setup
  test_grading.py             ← scaffold: grading tests (do not edit)
models/
  baseline_cnn.pt             ← pre-trained reference weights (committed)
  baseline_metrics.json       ← reference accuracy (committed)
  submission_cnn.pt           ← your trained model (commit after make train)
data/
  breastmnist.npz             ← dataset (auto-downloaded if missing)
requirements.txt
```

## How grading works on GitHub

When you push, GitHub Actions:

1. Checks out your repository.
2. Installs dependencies (`pip install -r requirements.txt`).
3. Runs `pytest tests/ -v`.

The tests load `models/submission_cnn.pt`, reconstruct your model using the
`BreastLesionCNN` class from `model.py`, and run inference on the BreastMNIST
test set. No training happens on the server — results are fast and
deterministic.

Score formula:

$$\text{score} = \text{clamp}\!\left(5 + 10 \cdot \frac{\text{accuracy} - \text{acc}_\text{ref}}{1 - \text{acc}_\text{ref}},\; 5,\; 15\right)$$


