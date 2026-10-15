import numpy as np

from src.metrics import bootstrap_accuracy_ci, classification_metrics, expected_calibration_error


def test_ece_zero_when_confidence_matches_accuracy():
    # 4 predictions at confidence 0.75, 3 correct -> perfectly calibrated
    probs = np.array([[0.75, 0.25]] * 4)
    labels = np.array([0, 0, 0, 1])
    assert expected_calibration_error(probs, labels) < 1e-9


def test_ece_overconfident():
    probs = np.array([[0.9, 0.1]] * 2)
    labels = np.array([0, 1])  # acc 0.5 at conf 0.9
    assert abs(expected_calibration_error(probs, labels) - 0.4) < 1e-9


def test_classification_metrics_basic():
    probs = np.eye(3)[[0, 1, 2, 2]]
    labels = np.array([0, 1, 2, 1])
    m = classification_metrics(probs, labels, ["a", "b", "c"], bootstrap=200)
    assert m["acc"] == 0.75
    assert m["per_class_recall"] == {"a": 1.0, "b": 0.5, "c": 1.0}
    lo, hi = m["acc_ci95"]
    assert 0 <= lo <= 0.75 <= hi <= 1


def test_macro_f1_ignores_classes_absent_from_split():
    probs = np.eye(3)[[0, 1]]
    m = classification_metrics(probs, np.array([0, 1]), ["a", "b", "c"])
    assert m["macro_f1"] == 1.0 and m["n_classes_present"] == 2


def test_bootstrap_deterministic():
    c = np.array([1, 0, 1, 1, 0, 1], float)
    assert bootstrap_accuracy_ci(c, 100, 0) == bootstrap_accuracy_ci(c, 100, 0)
