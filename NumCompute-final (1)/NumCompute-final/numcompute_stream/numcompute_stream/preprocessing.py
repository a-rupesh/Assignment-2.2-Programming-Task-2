"""Streaming-compatible preprocessing transformers for NumCompute Stream.

All classes extend the original NumCompute fit/transform contract with a
``partial_fit(X)`` method that updates statistics incrementally without
re-seeing past data.  They remain compatible with Pipeline from the base
package.

Classes
-------
StreamingStandardScaler
    Incremental z-score normalisation using Welford's online algorithm.
StreamingMinMaxScaler
    Incremental min/max scaling with running extreme tracking.
"""

from __future__ import annotations

import numpy as np


def _as_2d_float(X, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    return arr


def _check_features(X: np.ndarray, n_features_in_: int) -> None:
    if X.shape[1] != n_features_in_:
        raise ValueError(
            f"Input has {X.shape[1]} features; scaler was fitted with {n_features_in_}."
        )


class StreamingStandardScaler:
    """Incremental z-score scaler using Welford's online algorithm.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_samples_seen_ : int
        Total number of non-NaN samples processed per feature.
    mean_ : np.ndarray of shape (n_features,)
        Running column means.
    var_ : np.ndarray of shape (n_features,)
        Running column population variances.
    std_ : np.ndarray of shape (n_features,)
        Running column standard deviations (1.0 for constant columns).
    n_features_in_ : int
        Number of features seen during first partial_fit.

    Examples
    --------
    >>> scaler = StreamingStandardScaler()
    >>> scaler.partial_fit(np.array([[1., 2.], [3., 4.]]))
    >>> scaler.partial_fit(np.array([[5., 6.]]))
    >>> X_scaled = scaler.transform(np.array([[3., 4.]]))
    """

    def __init__(self) -> None:
        self.n_samples_seen_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self._M2: np.ndarray | None = None          # sum of squared deviations
        self.var_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise(self, n_features: int) -> None:
        self.n_features_in_ = n_features
        self.n_samples_seen_ = np.zeros(n_features, dtype=float)
        self.mean_ = np.zeros(n_features, dtype=float)
        self._M2 = np.zeros(n_features, dtype=float)
        self.var_ = np.zeros(n_features, dtype=float)
        self.std_ = np.ones(n_features, dtype=float)

    def _update_column(self, col_idx: int, values: np.ndarray) -> None:
        """Welford update for one feature column (ignores NaN)."""
        valid = values[~np.isnan(values)]
        for v in valid:
            self.n_samples_seen_[col_idx] += 1
            n = self.n_samples_seen_[col_idx]
            delta = v - self.mean_[col_idx]
            self.mean_[col_idx] += delta / n
            delta2 = v - self.mean_[col_idx]
            self._M2[col_idx] += delta * delta2

    def _recompute_derived(self) -> None:
        n = np.maximum(self.n_samples_seen_, 1.0)
        self.var_ = self._M2 / n
        std = np.sqrt(self.var_)
        self.std_ = np.where(std == 0, 1.0, std)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partial_fit(self, X) -> "StreamingStandardScaler":
        """Update running statistics with a new chunk of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New chunk of numeric data. NaN values are ignored.

        Returns
        -------
        self
        """
        X = _as_2d_float(X)
        if self.n_features_in_ is None:
            self._initialise(X.shape[1])
        else:
            _check_features(X, self.n_features_in_)

        for col in range(X.shape[1]):
            self._update_column(col, X[:, col])
        self._recompute_derived()
        return self

    def fit(self, X) -> "StreamingStandardScaler":
        """Reset and fit on a single batch (delegates to partial_fit)."""
        self.n_features_in_ = None
        return self.partial_fit(X)

    def transform(self, X) -> np.ndarray:
        """Standardise X using the running statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Standardised array; NaN values are preserved.
        """
        if self.mean_ is None:
            raise ValueError("Call partial_fit before transform.")
        X = _as_2d_float(X)
        _check_features(X, self.n_features_in_)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(X).transform(X)


class StreamingMinMaxScaler:
    """Incremental min-max scaler that tracks running extremes chunk-by-chunk.

    Parameters
    ----------
    feature_range : tuple of (float, float), default=(0.0, 1.0)
        Target range for scaled values.

    Attributes
    ----------
    data_min_ : np.ndarray of shape (n_features,)
        Running per-feature minimum.
    data_max_ : np.ndarray of shape (n_features,)
        Running per-feature maximum.
    n_features_in_ : int

    Notes
    -----
    Unlike batch MinMaxScaler, the range expands monotonically as more chunks
    arrive, so transform results may shift across chunks.

    Examples
    --------
    >>> scaler = StreamingMinMaxScaler()
    >>> scaler.partial_fit(np.array([[0., 10.], [5., 20.]]))
    >>> scaler.partial_fit(np.array([[-5., 30.]]))
    >>> scaler.transform(np.array([[0., 20.]]))
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)) -> None:
        low, high = feature_range
        if low >= high:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = (float(low), float(high))
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def _initialise(self, n_features: int) -> None:
        self.n_features_in_ = n_features
        self.data_min_ = np.full(n_features, np.inf)
        self.data_max_ = np.full(n_features, -np.inf)

    def partial_fit(self, X) -> "StreamingMinMaxScaler":
        """Update running min/max with a new chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _as_2d_float(X)
        if self.n_features_in_ is None:
            self._initialise(X.shape[1])
        else:
            _check_features(X, self.n_features_in_)

        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        self.data_min_ = np.minimum(self.data_min_, chunk_min)
        self.data_max_ = np.maximum(self.data_max_, chunk_max)
        return self

    def fit(self, X) -> "StreamingMinMaxScaler":
        """Reset and fit on a single batch."""
        self.n_features_in_ = None
        return self.partial_fit(X)

    def transform(self, X) -> np.ndarray:
        """Scale X to the target feature range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """
        if self.data_min_ is None:
            raise ValueError("Call partial_fit before transform.")
        X = _as_2d_float(X)
        _check_features(X, self.n_features_in_)
        low, high = self.feature_range
        data_range = np.where(
            self.data_max_ == self.data_min_, 1.0,
            self.data_max_ - self.data_min_
        )
        return (X - self.data_min_) / data_range * (high - low) + low

    def fit_transform(self, X) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(X).transform(X)
