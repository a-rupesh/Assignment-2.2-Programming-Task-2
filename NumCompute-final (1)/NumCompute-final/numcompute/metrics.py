"""Evaluation metrics for classification and regression."""

from __future__ import annotations

import numpy as np


def _validate_same_shape_1d(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return yt, yp


def _drop_nan_pairs(y_true, y_pred):
    if np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number):
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        return y_true[mask], y_pred[mask]
    return y_true, y_pred


def _validate_binary(y_true, y_pred):
    y_true, y_pred = _validate_same_shape_1d(y_true, y_pred)
    y_true, y_pred = _drop_nan_pairs(y_true.astype(float), y_pred.astype(float))
    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.all(np.isin(y_pred, [0.0, 1.0])):
        raise ValueError("y_pred must contain only binary labels 0 and 1.")
    return y_true.astype(int), y_pred.astype(int)


def accuracy(y_true, y_pred):
    """Return binary classification accuracy."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    """Return binary precision."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tp + fp
    return 0.0 if denom == 0 else float(tp / denom)


def recall(y_true, y_pred):
    """Return binary recall."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = tp + fn
    return 0.0 if denom == 0 else float(tp / denom)


def f1(y_true, y_pred):
    """Return binary F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return 0.0 if denom == 0 else float(2 * p * r / denom)


def confusion_matrix(y_true, y_pred):
    """
    Return confusion matrix in the standard layout:
    [[TN, FP],
     [FN, TP]]
    """
    y_true, y_pred = _validate_binary(y_true, y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def mse(y_true, y_pred):
    """Return mean squared error for regression."""
    y_true, y_pred = _validate_same_shape_1d(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    y_true, y_pred = _drop_nan_pairs(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true - y_pred) ** 2))

def roc_curve(y_true, y_score):
    """
    Compute ROC curve for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n,)
        Predicted scores or probabilities.

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    thresholds : np.ndarray
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays.")
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must be binary (0 and 1).")

    # sort by descending score
    desc_idx = np.argsort(-y_score)
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    # cumulative sums
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)

    # total positives / negatives
    P = np.sum(y_true)
    N = len(y_true) - P

    if P == 0 or N == 0:
        raise ValueError("ROC curve is undefined when only one class is present.")

    tpr = tp / P
    fpr = fp / N

    thresholds = y_score

    return fpr, tpr, thresholds

def auc(x, y):
    """
    Compute Area Under Curve using trapezoidal rule.

    Parameters
    ----------
    x : array-like
    y : array-like

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")

    return float(np.trapezoid(y, x))
