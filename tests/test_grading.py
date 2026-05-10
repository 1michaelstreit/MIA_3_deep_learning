"""Grading tests for the BreastMNIST deep learning assignment.

Scoring (15 points total):
  - 5 points  : reference CNN accuracy (default model.py + config.yaml, unchanged)
  - 15 points : test-set accuracy reaches 1.0
  - Linear interpolation between the two

Constraints that are also tested:
  - models/submission_cnn.pt must exist (commit it after running 'make train')
  - Model file must be ≤ 10 MB
  - Inference on the full test set must complete in ≤ 30 seconds

DO NOT MODIFY THIS FILE.

Check your score locally:
    make test          # or: pytest tests/ -v
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import (
    MODEL_SIZE_LIMIT_MB,
    INFERENCE_TIME_LIMIT_S,
    get_model_size_mb,
    load_model,
    run_inference,
)
from train import compute_score, load_baseline_accuracy, load_config, load_data

SUBMISSION_PATH = ROOT / "models" / "submission_cnn.pt"


@lru_cache(maxsize=1)
def _run_grading() -> dict:
    """Load and evaluate the submission model exactly once per test session."""
    from model import BreastLesionCNN

    model = load_model(BreastLesionCNN, SUBMISSION_PATH)

    config = load_config(str(ROOT / "config.yaml"))
    d_cfg = config["data"]
    data = load_data(
        dataset_name=d_cfg["dataset_name"],
        data_root=str(ROOT / d_cfg["data_root"].lstrip("./")),
    )

    accuracy, elapsed = run_inference(model, data["test_images"], data["test_labels"])
    size_mb = get_model_size_mb(SUBMISSION_PATH)
    baseline_accuracy = load_baseline_accuracy(
        str(ROOT / "models" / "baseline_metrics.json")
    )
    score = compute_score(accuracy, baseline_accuracy)

    print(f"\n[grading] accuracy={accuracy:.4f}")
    print(f"[grading] baseline={baseline_accuracy:.4f}")
    print(f"[grading] score={score:.2f} / 15")
    print(f"[grading] model_size={size_mb:.2f} MB")
    print(f"[grading] inference_time={elapsed:.3f} s")

    return {
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "score": score,
        "size_mb": size_mb,
        "inference_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Constraint tests (must all pass for any score to be awarded)
# ---------------------------------------------------------------------------


def test_model_file_exists() -> None:
    """Submission model must be present in the repository."""
    assert SUBMISSION_PATH.exists(), (
        f"{SUBMISSION_PATH.name} not found in models/.\n"
        "Run 'make train', then commit and push the file:\n"
        "  git add models/submission_cnn.pt && git commit -m 'Add trained model'"
    )


def test_model_size_within_limit() -> None:
    """Model file must not exceed the 10 MB size limit."""
    result = _run_grading()
    assert result["size_mb"] <= MODEL_SIZE_LIMIT_MB, (
        f"Model file is {result['size_mb']:.2f} MB — exceeds the "
        f"{MODEL_SIZE_LIMIT_MB:.0f} MB limit. Reduce the number of "
        "parameters and retrain."
    )


def test_inference_time_within_limit() -> None:
    """Inference on the full test set must complete within 30 seconds."""
    result = _run_grading()
    assert result["inference_time_s"] <= INFERENCE_TIME_LIMIT_S, (
        f"Inference took {result['inference_time_s']:.2f} s — exceeds the "
        f"{INFERENCE_TIME_LIMIT_S:.0f} s limit."
    )


# ---------------------------------------------------------------------------
# Score threshold tests  (30 × 0.5 pt = 15 pts max, max-score "15")
# Each function is a separate test so the autograding runner can find them
# via static analysis (parametrize is not supported by the runner).
# ---------------------------------------------------------------------------


def test_score_threshold_05() -> None:
    """Pass if the computed score is at least 0.5."""
    result = _run_grading()
    assert result["score"] >= 0.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 0.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_10() -> None:
    """Pass if the computed score is at least 1.0."""
    result = _run_grading()
    assert result["score"] >= 1.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 1.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_15() -> None:
    """Pass if the computed score is at least 1.5."""
    result = _run_grading()
    assert result["score"] >= 1.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 1.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_20() -> None:
    """Pass if the computed score is at least 2.0."""
    result = _run_grading()
    assert result["score"] >= 2.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 2.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_25() -> None:
    """Pass if the computed score is at least 2.5."""
    result = _run_grading()
    assert result["score"] >= 2.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 2.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_30() -> None:
    """Pass if the computed score is at least 3.0."""
    result = _run_grading()
    assert result["score"] >= 3.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 3.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_35() -> None:
    """Pass if the computed score is at least 3.5."""
    result = _run_grading()
    assert result["score"] >= 3.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 3.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_40() -> None:
    """Pass if the computed score is at least 4.0."""
    result = _run_grading()
    assert result["score"] >= 4.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 4.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_45() -> None:
    """Pass if the computed score is at least 4.5."""
    result = _run_grading()
    assert result["score"] >= 4.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 4.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_50() -> None:
    """Pass if the computed score is at least 5.0."""
    result = _run_grading()
    assert result["score"] >= 5.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 5.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_55() -> None:
    """Pass if the computed score is at least 5.5."""
    result = _run_grading()
    assert result["score"] >= 5.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 5.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_60() -> None:
    """Pass if the computed score is at least 6.0."""
    result = _run_grading()
    assert result["score"] >= 6.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 6.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_65() -> None:
    """Pass if the computed score is at least 6.5."""
    result = _run_grading()
    assert result["score"] >= 6.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 6.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_70() -> None:
    """Pass if the computed score is at least 7.0."""
    result = _run_grading()
    assert result["score"] >= 7.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 7.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_75() -> None:
    """Pass if the computed score is at least 7.5."""
    result = _run_grading()
    assert result["score"] >= 7.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 7.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_80() -> None:
    """Pass if the computed score is at least 8.0."""
    result = _run_grading()
    assert result["score"] >= 8.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 8.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_85() -> None:
    """Pass if the computed score is at least 8.5."""
    result = _run_grading()
    assert result["score"] >= 8.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 8.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_90() -> None:
    """Pass if the computed score is at least 9.0."""
    result = _run_grading()
    assert result["score"] >= 9.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 9.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_95() -> None:
    """Pass if the computed score is at least 9.5."""
    result = _run_grading()
    assert result["score"] >= 9.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 9.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_100() -> None:
    """Pass if the computed score is at least 10.0."""
    result = _run_grading()
    assert result["score"] >= 10.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 10.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_105() -> None:
    """Pass if the computed score is at least 10.5."""
    result = _run_grading()
    assert result["score"] >= 10.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 10.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_110() -> None:
    """Pass if the computed score is at least 11.0."""
    result = _run_grading()
    assert result["score"] >= 11.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 11.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_115() -> None:
    """Pass if the computed score is at least 11.5."""
    result = _run_grading()
    assert result["score"] >= 11.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 11.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_120() -> None:
    """Pass if the computed score is at least 12.0."""
    result = _run_grading()
    assert result["score"] >= 12.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 12.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_125() -> None:
    """Pass if the computed score is at least 12.5."""
    result = _run_grading()
    assert result["score"] >= 12.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 12.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_130() -> None:
    """Pass if the computed score is at least 13.0."""
    result = _run_grading()
    assert result["score"] >= 13.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 13.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_135() -> None:
    """Pass if the computed score is at least 13.5."""
    result = _run_grading()
    assert result["score"] >= 13.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 13.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_140() -> None:
    """Pass if the computed score is at least 14.0."""
    result = _run_grading()
    assert result["score"] >= 14.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 14.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_145() -> None:
    """Pass if the computed score is at least 14.5."""
    result = _run_grading()
    assert result["score"] >= 14.5, (
        f"Score {result['score']:.2f} / 15 is below threshold 14.5. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )


def test_score_threshold_150() -> None:
    """Pass if the computed score is at least 15.0."""
    result = _run_grading()
    assert result["score"] >= 15.0, (
        f"Score {result['score']:.2f} / 15 is below threshold 15.0. "
        f"accuracy={result['accuracy']:.4f}, reference={result['baseline_accuracy']:.4f}"
    )
