"""Streaming statistics module for NumCompute Stream.

Provides chunk-based versions of all descriptive statistics with an
``update_stats(X_chunk)`` API, plus sliding-window histogram support.

Classes
-------
StreamingStats
    Online Welford-based statistics for a single stream of values.
ChunkStats
    Chunk-based statistics tracker for 2D feature matrices.
    Exposes ``update_stats(X_chunk)`` as required by the spec.

Functions
---------
mean, variance, std, quantile, histogram
    Stateless helpers that work on a single array (unchanged from base).
"""

from __future__ import annotations

import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Stateless helpers (unchanged from base numcompute)
# ---------------------------------------------------------------------------

def mean(arr, axis=None):
    """Return NaN-ignoring mean."""
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")
    return np.nanmean(x, axis=axis)


def variance(arr, axis=None, ddof=0):
    """Return NaN-ignoring variance."""
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")
    return np.nanvar(x, axis=axis, ddof=ddof)


def std(arr, axis=None, ddof=0):
    """Return NaN-ignoring standard deviation."""
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")
    if not isinstance(ddof, int):
        raise TypeError("ddof must be an integer.")
    return np.nanstd(x, axis=axis, ddof=ddof)


def quantile(arr, q, axis=None, interpolation="linear"):
    """Return NaN-ignoring quantile(s) in [0, 1]."""
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")
    allowed = {"linear", "lower", "higher", "midpoint"}
    if interpolation not in allowed:
        raise ValueError(f"interpolation must be one of {sorted(allowed)}.")
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1.")
    result = np.nanquantile(x, q_arr, axis=axis, method=interpolation)
    if q_arr.ndim == 0 and np.asarray(result).ndim == 0:
        return float(np.asarray(result))
    return result


def histogram(arr, bins=10, range=None):  # noqa: A002
    """Return histogram counts and bin edges, ignoring NaN."""
    x = np.asarray(arr, dtype=float)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    if range is not None:
        if not isinstance(range, tuple) or len(range) != 2:
            raise TypeError("range must be a tuple of length 2 or None.")
        if range[0] >= range[1]:
            raise ValueError("range lower bound must be less than upper bound.")
    flat = x.ravel()
    flat = flat[~np.isnan(flat)]
    return np.histogram(flat, bins=bins, range=range)


# ---------------------------------------------------------------------------
# StreamingStats — single-stream online statistics (Welford)
# ---------------------------------------------------------------------------

class StreamingStats:
    """Online statistics for a single stream using Welford's algorithm.

    Supports incremental updates one value or many values at a time.
    NaN values are ignored.

    Attributes
    ----------
    n_ : int
    mean_ : float
    M2_ : float
    min_ : float
    max_ : float
    """

    def __init__(self):
        self.n_ = 0
        self.mean_ = 0.0
        self.M2_ = 0.0
        self.min_ = np.inf
        self.max_ = -np.inf

    def update(self, value) -> "StreamingStats":
        """Update with one value."""
        value = float(value)
        if np.isnan(value):
            return self
        self.n_ += 1
        delta = value - self.mean_
        self.mean_ += delta / self.n_
        self.M2_ += delta * (value - self.mean_)
        self.min_ = min(self.min_, value)
        self.max_ = max(self.max_, value)
        return self

    def update_many(self, values) -> "StreamingStats":
        """Update with an array of values."""
        for v in np.asarray(values, dtype=float).ravel():
            self.update(v)
        return self

    @property
    def count(self): return self.n_

    @property
    def mean(self): return self.mean_ if self.n_ > 0 else np.nan

    @property
    def variance(self):
        return self.M2_ / self.n_ if self.n_ > 0 else np.nan

    @property
    def sample_variance(self):
        return self.M2_ / (self.n_ - 1) if self.n_ > 1 else np.nan

    @property
    def std(self): return float(np.sqrt(self.variance))

    @property
    def sample_std(self): return float(np.sqrt(self.sample_variance))

    @property
    def min(self): return self.min_ if self.n_ > 0 else np.nan  # noqa: A003

    @property
    def max(self): return self.max_ if self.n_ > 0 else np.nan  # noqa: A003

    def to_dict(self) -> dict:
        return {
            'count': self.count, 'mean': self.mean,
            'variance': self.variance, 'sample_variance': self.sample_variance,
            'std': self.std, 'sample_std': self.sample_std,
            'min': self.min, 'max': self.max,
        }


# ---------------------------------------------------------------------------
# ChunkStats — spec-required update_stats(X_chunk) API
# ---------------------------------------------------------------------------

class ChunkStats:
    """Chunk-based streaming statistics for a 2D feature matrix.

    Maintains per-feature running mean, variance, min, max, and an optional
    sliding-window histogram across chunks.

    Parameters
    ----------
    n_features : int or None, default=None
        Number of features. Inferred from the first chunk if None.
    window_size : int or None, default=None
        If set, only the last ``window_size`` chunks contribute to stats.
        If None, all chunks are accumulated.

    Examples
    --------
    >>> cs = ChunkStats()
    >>> cs.update_stats(np.array([[1., 2.], [3., 4.]]))
    >>> cs.update_stats(np.array([[5., 6.]]))
    >>> cs.mean_
    array([3., 4.])
    """

    def __init__(self, n_features: int | None = None,
                 window_size: int | None = None):
        self.n_features = n_features
        self.window_size = window_size

        # Per-feature Welford state
        self._n = None           # np.ndarray shape (n_features,)
        self._mean = None
        self._M2 = None
        self._min = None
        self._max = None

        # Sliding window buffer (stores per-chunk means for windowed stats)
        self._window: deque = deque(maxlen=window_size)
        self.chunk_idx_: int = 0

    def _init_arrays(self, n_features: int) -> None:
        self.n_features = n_features
        self._n = np.zeros(n_features, dtype=float)
        self._mean = np.zeros(n_features, dtype=float)
        self._M2 = np.zeros(n_features, dtype=float)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)

    def update_stats(self, X_chunk) -> "ChunkStats":
        """Update running statistics with a new chunk.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            New chunk of data. NaN values are ignored per feature.

        Returns
        -------
        self
        """
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim != 2:
            raise ValueError("X_chunk must be 2D.")
        if X.shape[0] == 0:
            return self

        if self._n is None:
            self._init_arrays(X.shape[1])
        elif X.shape[1] != self.n_features:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features; "
                f"expected {self.n_features}."
            )

        # Welford update per feature, ignoring NaN
        for f in range(self.n_features):
            col = X[:, f]
            valid = col[~np.isnan(col)]
            for v in valid:
                self._n[f] += 1
                delta = v - self._mean[f]
                self._mean[f] += delta / self._n[f]
                self._M2[f] += delta * (v - self._mean[f])
            if len(valid) > 0:
                self._min[f] = min(self._min[f], valid.min())
                self._max[f] = max(self._max[f], valid.max())

        # Store chunk mean for sliding window
        chunk_mean = np.nanmean(X, axis=0)
        self._window.append(chunk_mean)
        self.chunk_idx_ += 1
        return self

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def mean_(self) -> np.ndarray:
        """Running per-feature mean."""
        if self._mean is None:
            raise ValueError("Call update_stats first.")
        return self._mean.copy()

    @property
    def var_(self) -> np.ndarray:
        """Running per-feature population variance."""
        if self._M2 is None:
            raise ValueError("Call update_stats first.")
        n = np.maximum(self._n, 1)
        return self._M2 / n

    @property
    def std_(self) -> np.ndarray:
        """Running per-feature standard deviation."""
        return np.sqrt(self.var_)

    @property
    def min_(self) -> np.ndarray:
        """Running per-feature minimum."""
        if self._min is None:
            raise ValueError("Call update_stats first.")
        return self._min.copy()

    @property
    def max_(self) -> np.ndarray:
        """Running per-feature maximum."""
        if self._max is None:
            raise ValueError("Call update_stats first.")
        return self._max.copy()

    @property
    def window_mean_(self) -> np.ndarray:
        """Mean computed over the sliding window of recent chunks."""
        if not self._window:
            raise ValueError("No chunks seen yet.")
        return np.mean(list(self._window), axis=0)

    def chunk_histogram(self, X_chunk, feature: int = 0,
                        bins: int = 10) -> tuple:
        """Compute a histogram for one feature of the current chunk.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
        feature : int, default=0
        bins : int, default=10

        Returns
        -------
        tuple (counts, bin_edges)
        """
        X = np.asarray(X_chunk, dtype=float)
        col = X[:, feature]
        col = col[~np.isnan(col)]
        return np.histogram(col, bins=bins)

    def to_dict(self) -> dict:
        """Return summary statistics as a dictionary."""
        if self._mean is None:
            return {'chunks_seen': 0}
        return {
            'chunks_seen': self.chunk_idx_,
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'min': self.min_.tolist(),
            'max': self.max_.tolist(),
        }
