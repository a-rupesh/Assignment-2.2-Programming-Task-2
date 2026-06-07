"""StreamTrainer — central coordinator for streaming ML pipelines.

Manages a preprocessing pipeline + model, logs per-chunk metrics,
tracks memory footprint, and maintains cumulative accuracy.

Classes
-------
StreamTrainer
    Wraps a scaler and a streaming model. Call fit_chunk(X, y) for each
    incoming chunk and score_chunk(X, y) to evaluate.

Example
-------
    from numcompute_stream.stream import StreamTrainer
    from numcompute_stream.preprocessing import StreamingStandardScaler
    from numcompute_stream.tree import DecisionTreeClassifier

    trainer = StreamTrainer(
        model=DecisionTreeClassifier(min_samples_split=20),
        scaler=StreamingStandardScaler(),
    )
    for X_chunk, y_chunk in data_stream:
        trainer.fit_chunk(X_chunk, y_chunk)
        metrics = trainer.score_chunk(X_chunk, y_chunk)
        print(metrics)

    print(trainer.get_log())
"""

from __future__ import annotations

import sys
import time
import numpy as np


def _memory_bytes(obj) -> int:
    """Rough memory estimate of an object in bytes."""
    return sys.getsizeof(obj)


class StreamTrainer:
    """Manages a streaming model and preprocessing pipeline with metric logging.

    Parameters
    ----------
    model : object
        A streaming classifier with ``partial_fit(X, y)`` and ``predict(X)``.
    scaler : object or None, default=None
        A streaming scaler with ``partial_fit(X)`` and ``transform(X)``.
        If None, raw features are passed directly to the model.
    classes : array-like or None, default=None
        All possible class labels. Required on first chunk if not inferable.

    Attributes
    ----------
    log_ : list of dict
        Per-chunk log entries. Each entry contains:
        - chunk_idx : int
        - n_samples : int
        - accuracy : float
        - cumulative_accuracy : float
        - fit_time_s : float
        - memory_bytes : int
        - timestamp : float
    chunk_idx_ : int
        Number of chunks processed so far.
    """

    def __init__(self, model, scaler=None, classes=None):
        if not hasattr(model, 'partial_fit'):
            raise ValueError("model must implement partial_fit().")
        if scaler is not None and not hasattr(scaler, 'partial_fit'):
            raise ValueError("scaler must implement partial_fit().")

        self.model = model
        self.scaler = scaler
        self.classes = classes
        self.log_: list[dict] = []
        self.chunk_idx_: int = 0
        self._total_correct: int = 0
        self._total_seen: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale X, optionally updating the scaler first."""
        if self.scaler is None:
            return X
        if fit:
            self.scaler.partial_fit(X)
        return self.scaler.transform(X)

    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_chunk(self, X, y) -> "StreamTrainer":
        """Train the model on one incoming chunk.

        Updates the scaler (if any) and the model incrementally.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        t0 = time.perf_counter()
        X_proc = self._preprocess(X, fit=True)

        kwargs = {}
        if self.classes is not None:
            kwargs['classes'] = np.asarray(self.classes)

        self.model.partial_fit(X_proc, y, **kwargs)
        fit_time = time.perf_counter() - t0

        # Score on the chunk just trained (training accuracy)
        y_pred = self.model.predict(X_proc)
        acc = self._compute_accuracy(y, y_pred)

        self._total_correct += int(np.sum(y == y_pred))
        self._total_seen += len(y)
        cumulative_acc = (self._total_correct / self._total_seen
                          if self._total_seen > 0 else 0.0)

        self.log_.append({
            'chunk_idx': self.chunk_idx_,
            'n_samples': len(y),
            'accuracy': acc,
            'cumulative_accuracy': cumulative_acc,
            'fit_time_s': round(fit_time, 6),
            'memory_bytes': _memory_bytes(self.model),
            'timestamp': time.time(),
        })

        self.chunk_idx_ += 1
        return self

    def score_chunk(self, X, y) -> dict:
        """Evaluate the current model on a chunk without updating it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        dict with keys: accuracy, n_samples, chunk_idx
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X_proc = self._preprocess(X, fit=False)
        y_pred = self.model.predict(X_proc)
        acc = self._compute_accuracy(y, y_pred)
        return {
            'chunk_idx': self.chunk_idx_,
            'n_samples': len(y),
            'accuracy': acc,
        }

    def get_log(self) -> list[dict]:
        """Return the full per-chunk metric log.

        Returns
        -------
        list of dict
            One entry per chunk processed by fit_chunk().
        """
        return self.log_

    def accuracy_history(self) -> list[float]:
        """Return per-chunk accuracy values as a plain list."""
        return [entry['accuracy'] for entry in self.log_]

    def cumulative_accuracy_history(self) -> list[float]:
        """Return cumulative accuracy values as a plain list."""
        return [entry['cumulative_accuracy'] for entry in self.log_]

    def reset_log(self) -> "StreamTrainer":
        """Clear the metric log and reset counters."""
        self.log_ = []
        self.chunk_idx_ = 0
        self._total_correct = 0
        self._total_seen = 0
        return self

    def summary(self) -> dict:
        """Return a summary of training so far.

        Returns
        -------
        dict with keys:
            chunks_processed, total_samples, final_accuracy,
            cumulative_accuracy, total_fit_time_s
        """
        if not self.log_:
            return {'chunks_processed': 0}
        return {
            'chunks_processed': self.chunk_idx_,
            'total_samples': self._total_seen,
            'final_accuracy': self.log_[-1]['accuracy'],
            'cumulative_accuracy': self.log_[-1]['cumulative_accuracy'],
            'total_fit_time_s': round(
                sum(e['fit_time_s'] for e in self.log_), 4),
        }
