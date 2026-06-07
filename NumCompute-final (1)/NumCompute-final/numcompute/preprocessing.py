"""Preprocessing utilities for NumCompute.

This module provides transformer-style preprocessing classes inspired by common
machine-learning workflows. Each transformer implements:

- fit(X) -> self
- transform(X) -> transformed array
- fit_transform(X) -> transformed array

The implementations use NumPy arrays and avoid element-wise Python loops.
"""

from __future__ import annotations

import numpy as np


def _as_2d_array(X, *, dtype=None, name="X") -> np.ndarray:
    """Convert input to a 2D NumPy array."""
    arr = np.asarray(X, dtype=dtype)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")

    return arr


def _check_feature_count(X: np.ndarray, n_features_in_: int) -> None:
    """Validate that transform data has the same number of features as fit data."""
    if X.shape[1] != n_features_in_:
        raise ValueError(
            f"Input has {X.shape[1]} features, but transformer was fitted with "
            f"{n_features_in_} features."
        )


def _check_numeric_feature_range(feature_range):
    """Validate and return a numeric feature range tuple."""
    if not isinstance(feature_range, tuple) or len(feature_range) != 2:
        raise ValueError("feature_range must be a length-2 tuple.")

    low, high = feature_range

    if not np.isscalar(low) or not np.isscalar(high):
        raise TypeError("feature_range values must be numeric scalars.")

    low = float(low)
    high = float(high)

    if low >= high:
        raise ValueError("feature_range must satisfy min < max.")

    return low, high


class StandardScaler:
    """
    Standardize features using z-score normalization.

    Each feature is transformed as:

        z = (x - mean) / standard_deviation

    NaN values are ignored during fitting and preserved during transformation.
    Constant columns are safely handled by replacing zero standard deviations
    with 1.0.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Fit the scaler by estimating column-wise mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric input data.

        Returns
        -------
        StandardScaler
            The fitted scaler.
        """
        X = _as_2d_array(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)

        self.std_ = np.where((self.std_ == 0) | np.isnan(self.std_), 1.0, self.std_)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Standardize input data using fitted statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric input data.

        Returns
        -------
        np.ndarray
            Standardized data.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before transform().")

        X = _as_2d_array(X, dtype=float)
        _check_feature_count(X, self.n_features_in_)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit the scaler and return transformed data."""
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scale features to a specified numeric range.

    Each feature is transformed into the chosen feature_range. NaN values are
    ignored during fitting and preserved during transformation. Constant columns
    are safely mapped to the lower bound of the target range.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = _check_numeric_feature_range(feature_range)
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Fit the scaler by estimating column-wise min and max.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric input data.

        Returns
        -------
        MinMaxScaler
            The fitted scaler.
        """
        X = _as_2d_array(X, dtype=float)

        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_min_ = np.where(np.isnan(self.data_min_), 0.0, self.data_min_)
        self.data_max_ = np.where(np.isnan(self.data_max_), 0.0, self.data_max_)

        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Scale input data using fitted min/max statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric input data.

        Returns
        -------
        np.ndarray
            Scaled data.
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("MinMaxScaler must be fitted before transform().")

        X = _as_2d_array(X, dtype=float)
        _check_feature_count(X, self.n_features_in_)

        low, high = self.feature_range
        X_std = (X - self.data_min_) / self.data_range_
        return X_std * (high - low) + low

    def fit_transform(self, X):
        """Fit the scaler and return transformed data."""
        return self.fit(X).transform(X)


class SimpleImputer:
    """
    Replace missing numeric values column-wise.

    Supported strategies:
    - "mean": replace NaNs with column mean
    - "median": replace NaNs with column median
    - "constant": replace NaNs with fill_value

    If a column is entirely NaN and strategy is "mean" or "median", fill_value
    is used for that column.
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        allowed = {"mean", "median", "constant"}

        if strategy not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Fit the imputer by estimating replacement values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric input data.

        Returns
        -------
        SimpleImputer
            The fitted imputer.
        """
        X = _as_2d_array(X, dtype=float)

        if self.strategy == "mean":
            statistics = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            statistics = np.nanmedian(X, axis=0)
        else:
            statistics = np.full(X.shape[1], self.fill_value, dtype=float)

        self.statistics_ = np.where(np.isnan(statistics), float(self.fill_value), statistics)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Replace NaN values using fitted statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric input data.

        Returns
        -------
        np.ndarray
            Data with NaNs replaced.
        """
        if self.statistics_ is None:
            raise ValueError("SimpleImputer must be fitted before transform().")

        X = _as_2d_array(X, dtype=float)
        _check_feature_count(X, self.n_features_in_)

        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X):
        """Fit the imputer and return transformed data."""
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    One-hot encode categorical columns into a dense integer matrix.

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default="error"
        How to handle categories observed during transform but not fit.
        - "error": raise ValueError
        - "ignore": output all zeros for unknown categories
    dtype : data-type, default=int
        Output dtype.

    Notes
    -----
    A small loop over feature columns is used because each categorical column
    can have a different number of categories. The comparisons themselves are
    vectorized across all samples.
    """

    def __init__(self, handle_unknown="error", dtype=int):
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        self.handle_unknown = handle_unknown
        self.dtype = dtype
        self.categories_ = None
        self.feature_indices_ = None
        self.n_features_in_ = None
        self.n_output_features_ = None

    def fit(self, X):
        """
        Learn sorted unique categories for each input column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical input data.

        Returns
        -------
        OneHotEncoder
            The fitted encoder.
        """
        X = _as_2d_array(X, dtype=object)

        categories = []
        sizes = []

        for i in range(X.shape[1]):
            cats = np.unique(X[:, i])
            categories.append(cats)
            sizes.append(cats.size)

        self.categories_ = categories
        self.feature_indices_ = np.concatenate(([0], np.cumsum(sizes)))
        self.n_features_in_ = X.shape[1]
        self.n_output_features_ = int(self.feature_indices_[-1])

        return self

    def transform(self, X):
        """
        Transform categorical data into one-hot encoded columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical input data.

        Returns
        -------
        np.ndarray
            Dense one-hot encoded matrix of shape
            (n_samples, total_number_of_categories).
        """
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transform().")

        X = _as_2d_array(X, dtype=object)
        _check_feature_count(X, self.n_features_in_)

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_output_features_), dtype=self.dtype)
        rows = np.arange(n_samples)

        for i, cats in enumerate(self.categories_):
            col = X[:, i]
            matches = col[:, None] == cats[None, :]
            known = np.any(matches, axis=1)

            if self.handle_unknown == "error" and np.any(~known):
                raise ValueError(f"Unknown category found in column {i}.")

            category_positions = np.argmax(matches, axis=1)
            valid_rows = rows[known]
            encoded_col_positions = self.feature_indices_[i] + category_positions[known]
            encoded[valid_rows, encoded_col_positions] = 1

        return encoded

    def fit_transform(self, X):
        """Fit the encoder and return encoded data."""
        return self.fit(X).transform(X)
