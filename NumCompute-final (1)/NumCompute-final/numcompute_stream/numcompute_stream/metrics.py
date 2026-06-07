"""Streaming classification and regression metrics for NumCompute Stream.

All metrics support incremental updates via ``update(y_true, y_pred)``
and expose ``reset()`` / ``result()`` methods as required by the spec.

Classes
-------
StreamingMetrics
    Accumulates accuracy, precision, recall, F1, confusion matrix, and AUC
    over an unbounded or rolling-window stream.

Functions
---------
accuracy, precision, recall, f1, confusion_matrix, mse
    Stateless helpers for single-chunk evaluation.
"""

from __future__ import annotations

import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def _check_binary_1d(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask].astype(int), yp[mask].astype(int)
    if not np.all(np.isin(yt, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.all(np.isin(yp, [0, 1])):
        raise ValueError("y_pred must contain only binary labels 0 and 1.")
    return yt, yp


def accuracy(y_true, y_pred) -> float:
    yt, yp = _check_binary_1d(y_true, y_pred)
    return float(np.mean(yt == yp)) if len(yt) > 0 else 0.0


def precision(y_true, y_pred) -> float:
    yt, yp = _check_binary_1d(y_true, y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred) -> float:
    yt, yp = _check_binary_1d(y_true, y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    yt, yp = _check_binary_1d(y_true, y_pred)
    return np.array([
        [np.sum((yt == 0) & (yp == 0)), np.sum((yt == 0) & (yp == 1))],
        [np.sum((yt == 1) & (yp == 0)), np.sum((yt == 1) & (yp == 1))],
    ], dtype=int)


def mse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    return float(np.mean((yt - yp) ** 2)) if len(yt) > 0 else 0.0


# ---------------------------------------------------------------------------
# StreamingMetrics — spec-required update / reset / result API
# ---------------------------------------------------------------------------

class StreamingMetrics:
    """Accumulates classification metrics over a streaming sequence of chunks.

    Supports both full-history accumulation and rolling-window metrics.

    Parameters
    ----------
    window_size : int or None, default=None
        If set, rolling-window metrics are computed over the last
        ``window_size`` chunks only. Full-history metrics always accumulate.
    n_classes : int, default=2
        Number of classes. Only binary (n_classes=2) is fully supported.

    Examples
    --------
    >>> sm = StreamingMetrics()
    >>> sm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    >>> sm.update(np.array([1, 0, 1, 1]), np.array([1, 0, 1, 0]))
    >>> sm.result()
    {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
    >>> sm.reset()
    """

    def __init__(self, window_size: int | None = None, n_classes: int = 2):
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        self.window_size = window_size
        self.n_classes = n_classes
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialise all accumulators."""
        # Full-history confusion matrix
        self._cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        self._total_samples = 0
        self._chunk_count = 0

        # Per-chunk metric log (for rolling window)
        self._window: deque = deque(maxlen=self.window_size)

        # AUC accumulation (trapezoidal over per-chunk scores)
        self._fpr_history: list[float] = []
        self._tpr_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, y_true, y_pred, y_score=None) -> "StreamingMetrics":
        """Update accumulators with one chunk of predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels.
        y_score : array-like of shape (n_samples,) or None
            Predicted probabilities for the positive class.
            Required for AUC tracking.

        Returns
        -------
        self
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        # Drop NaN pairs
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt, yp = yt[mask].astype(int), yp[mask].astype(int)

        if len(yt) == 0:
            return self

        # Update confusion matrix
        for true, pred in zip(yt, yp):
            if 0 <= true < self.n_classes and 0 <= pred < self.n_classes:
                self._cm[true, pred] += 1

        self._total_samples += len(yt)
        self._chunk_count += 1

        # Per-chunk metrics for rolling window
        chunk_acc = float(np.mean(yt == yp))
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        self._window.append({
            'accuracy': chunk_acc,
            'precision': prec,
            'recall': rec,
            'f1': f1_,
            'n_samples': len(yt),
        })

        # AUC: track per-chunk FPR/TPR if scores provided
        if y_score is not None:
            ys = np.asarray(y_score, dtype=float)[mask]
            P = yt.sum()
            N = len(yt) - P
            if P > 0 and N > 0:
                desc = np.argsort(-ys)
                tp_c = np.cumsum(yt[desc])
                fp_c = np.cumsum(1 - yt[desc])
                self._fpr_history.append(float(fp_c[-1] / N))
                self._tpr_history.append(float(tp_c[-1] / P))

        return self

    def reset(self) -> "StreamingMetrics":
        """Reset all accumulators to initial state.

        Returns
        -------
        self
        """
        self._reset_state()
        return self

    def result(self) -> dict:
        """Return current accumulated metrics.

        Returns
        -------
        dict with keys:
            accuracy, precision, recall, f1,
            confusion_matrix, total_samples, chunk_count,
            rolling_accuracy (if window_size set)
        """
        cm = self._cm
        total = self._total_samples

        if total == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0,
                'recall': 0.0, 'f1': 0.0,
                'confusion_matrix': cm.tolist(),
                'total_samples': 0,
                'chunk_count': 0,
            }

        acc = float(np.trace(cm) / total)

        # Binary metrics from cumulative confusion matrix
        if self.n_classes == 2:
            tp = int(cm[1, 1])
            fp = int(cm[0, 1])
            fn = int(cm[1, 0])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_  = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        else:
            prec = rec = f1_ = 0.0   # multi-class not fully supported

        out = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1_,
            'confusion_matrix': cm.tolist(),
            'total_samples': total,
            'chunk_count': self._chunk_count,
        }

        # Rolling window metrics
        if self._window:
            w = list(self._window)
            out['rolling_accuracy']  = float(np.mean([e['accuracy']  for e in w]))
            out['rolling_precision'] = float(np.mean([e['precision'] for e in w]))
            out['rolling_recall']    = float(np.mean([e['recall']    for e in w]))
            out['rolling_f1']        = float(np.mean([e['f1']        for e in w]))

        # AUC (trapezoidal over per-chunk TPR/FPR)
        if len(self._fpr_history) >= 2:
            fpr = np.array([0.0] + self._fpr_history)
            tpr = np.array([0.0] + self._tpr_history)
            out['auc'] = float(np.trapezoid(tpr, fpr))

        return out

    def accuracy_history(self) -> list[float]:
        """Return per-chunk accuracy as a list."""
        return [e['accuracy'] for e in self._window]

    def confusion_matrix_accumulated(self) -> np.ndarray:
        """Return the full accumulated confusion matrix."""
        return self._cm.copy()
