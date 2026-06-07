import numpy as np
import pytest

from numcompute.metrics import accuracy, auc, confusion_matrix, f1, mse, precision, recall, roc_curve

def test_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0]))

def test_nan_handling():
    y_true = np.array([1, np.nan, 0])
    y_pred = np.array([1, 1, 0])
    assert accuracy(y_true, y_pred) == 1.0

def test_empty_input():
    assert accuracy(np.array([]), np.array([])) == 0.0

def test_non_1d_input_raises():
    import pytest
    with pytest.raises(ValueError):
        accuracy(np.array([[0, 1]]), np.array([[0, 1]]))        

def test_accuracy_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_half_correct():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.5


def test_precision_recall_f1_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert precision(y_true, y_pred) == 0.5
    assert recall(y_true, y_pred) == 0.5
    assert f1(y_true, y_pred) == 0.5


def test_precision_zero_division_case():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert precision(y_true, y_pred) == 0.0


def test_recall_zero_division_case():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert recall(y_true, y_pred) == 0.0


def test_confusion_matrix_layout():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert np.array_equal(confusion_matrix(y_true, y_pred), np.array([[1, 1], [1, 1]]))


def test_mse_matches_numpy():
    y_true = np.array([3.5, 2.0, -1.0, 4.0])
    y_pred = np.array([3.0, 2.5, -2.0, 5.0])
    assert np.isclose(mse(y_true, y_pred), np.mean((y_true - y_pred) ** 2))


def test_invalid_non_binary_raises():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 2, 1]), np.array([0, 1, 1]))

def test_roc_curve_basic():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, _ = roc_curve(y_true, y_score)

    assert len(fpr) == len(tpr)
    assert np.all(fpr >= 0) and np.all(fpr <= 1)
    assert np.all(tpr >= 0) and np.all(tpr <= 1)


def test_auc_basic():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.75, 1.0])

    assert np.isclose(auc(x, y), np.trapezoid(y, x))        
