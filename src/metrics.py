"""Classification metrics: accuracy, macro-F1, per-class recall, ECE, bootstrap CI."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-1 ECE with equal-width confidence bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def bootstrap_accuracy_ci(correct: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI for accuracy. PlantDoc test is small, so this matters."""
    if n <= 0 or len(correct) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n, len(correct)))
    accs = correct[idx].mean(1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def classification_metrics(
    probs: np.ndarray, labels: np.ndarray, class_names: list[str], n_bins: int = 15, bootstrap: int = 0
) -> dict:
    preds = probs.argmax(1)
    present = np.unique(labels)  # macro averages over classes that actually occur in this split
    correct = (preds == labels).astype(float)
    recalls = recall_score(labels, preds, labels=present, average=None, zero_division=0)
    lo, hi = bootstrap_accuracy_ci(correct, bootstrap)
    return {
        "n": int(len(labels)),
        "n_classes_present": int(len(present)),
        "acc": float(correct.mean()),
        "acc_ci95": [lo, hi],
        "macro_f1": float(f1_score(labels, preds, labels=present, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "per_class_recall": {class_names[c]: float(r) for c, r in zip(present, recalls)},
        "confusion": confusion_matrix(labels, preds, labels=np.arange(len(class_names))).tolist(),
    }
