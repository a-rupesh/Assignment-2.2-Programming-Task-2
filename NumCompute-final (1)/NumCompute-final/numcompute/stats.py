"""Descriptive and streaming statistics for NumCompute.

This module provides NumPy-based descriptive statistics with NaN handling,
axis-wise operations, histograms, quantiles, and streaming statistics using
Welford's online algorithm.
"""

from __future__ import annotations

import numpy as np


def _validate_array(arr) -> np.ndarray:
    """Convert input to a floating NumPy array and validate dimensionality."""
    x = np.asarray(arr, dtype=float)

    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")

    return x


def _validate_axis(x: np.ndarray, axis):
    """Validate axis argument for an array."""
    if axis is None:
        return None

    if not isinstance(axis, int):
        raise TypeError("axis must be an integer or None.")

    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError(f"axis must be between {-x.ndim} and {x.ndim - 1}.")

    return axis


def mean(arr, axis=None):
    """
    Return the arithmetic mean, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None, default=None
        Axis along which to compute the mean. If None, the array is flattened.

    Returns
    -------
    float or np.ndarray
        Mean value or axis-wise means.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmean(x, axis=axis)


def median(arr, axis=None):
    """
    Return the median, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None, default=None
        Axis along which to compute the median. If None, the array is flattened.

    Returns
    -------
    float or np.ndarray
        Median value or axis-wise medians.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmedian(x, axis=axis)


def std(arr, axis=None, ddof=0):
    """
    Return the standard deviation, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None, default=None
        Axis along which to compute standard deviation.
    ddof : int, default=0
        Delta degrees of freedom. Use ddof=1 for sample standard deviation.

    Returns
    -------
    float or np.ndarray
        Standard deviation value or axis-wise standard deviations.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)

    if not isinstance(ddof, int):
        raise TypeError("ddof must be an integer.")

    return np.nanstd(x, axis=axis, ddof=ddof)


def min(arr, axis=None):  # noqa: A001
    """
    Return the minimum value, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None, default=None
        Axis along which to compute the minimum.

    Returns
    -------
    float or np.ndarray
        Minimum value or axis-wise minimum values.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmin(x, axis=axis)


def max(arr, axis=None):  # noqa: A001
    """
    Return the maximum value, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None, default=None
        Axis along which to compute the maximum.

    Returns
    -------
    float or np.ndarray
        Maximum value or axis-wise maximum values.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmax(x, axis=axis)


def histogram(arr, bins=10, range=None):  # noqa: A002
    """
    Compute histogram counts and bin edges after removing NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    bins : int, default=10
        Number of histogram bins.
    range : tuple or None, default=None
        Optional lower and upper range for bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Histogram counts and bin edges.
    """
    x = _validate_array(arr)

    if not isinstance(bins, int):
        raise TypeError("bins must be an integer.")

    if bins < 1:
        raise ValueError("bins must be positive.")

    if range is not None:
        if not isinstance(range, tuple) or len(range) != 2:
            raise TypeError("range must be a tuple of length 2 or None.")
        if range[0] >= range[1]:
            raise ValueError("range lower bound must be less than upper bound.")

    flat = x.ravel()
    flat = flat[~np.isnan(flat)]

    return np.histogram(flat, bins=bins, range=range)


def quantile(arr, q, axis=None, interpolation="linear"):
    """
    Compute quantiles in [0, 1], ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    q : float or array-like
        Quantile or quantiles to compute. Values must be between 0 and 1.
    axis : int or None, default=None
        Axis along which to compute quantiles.
    interpolation : {"linear", "lower", "higher", "midpoint"}, default="linear"
        Interpolation method.

    Returns
    -------
    float or np.ndarray
        Quantile value or axis-wise quantile values.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)

    allowed = {"linear", "lower", "higher", "midpoint"}
    if interpolation not in allowed:
        raise ValueError(f"interpolation must be one of {sorted(allowed)}.")

    q_arr = np.asarray(q, dtype=float)
    scalar = q_arr.ndim == 0

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1.")

    result = np.nanquantile(x, q_arr, axis=axis, method=interpolation)

    if scalar and np.asarray(result).ndim == 0:
        return float(np.asarray(result))

    return result


class StreamingStats:
    """
    Online descriptive statistics using Welford's algorithm.

    This class updates statistics one value at a time without storing the full
    dataset. NaN values are ignored.

    Attributes
    ----------
    n_ : int
        Number of non-NaN observations processed.
    mean_ : float
        Running mean.
    M2_ : float
        Running sum of squared deviations from the mean.
    min_ : float
        Minimum observed non-NaN value.
    max_ : float
        Maximum observed non-NaN value.
    """

    def __init__(self):
        self.n_ = 0
        self.mean_ = 0.0
        self.M2_ = 0.0
        self.min_ = np.inf
        self.max_ = -np.inf

    def update(self, value):
        """
        Update the running statistics with one value.

        Parameters
        ----------
        value : float
            New observation. NaN values are ignored.

        Returns
        -------
        StreamingStats
            The fitted object itself.
        """
        value = float(value)

        if np.isnan(value):
            return self

        self.n_ += 1

        delta = value - self.mean_
        self.mean_ += delta / self.n_
        delta2 = value - self.mean_
        self.M2_ += delta * delta2

        if value < self.min_:
            self.min_ = value

        if value > self.max_:
            self.max_ = value

        return self

    def update_many(self, values):
        """
        Update the running statistics with multiple values.

        Parameters
        ----------
        values : array-like
            Values to add to the running statistics.

        Returns
        -------
        StreamingStats
            The fitted object itself.
        """
        values = np.asarray(values, dtype=float).ravel()

        for value in values:
            self.update(value)

        return self

    @property
    def count(self):
        """Return the number of non-NaN observations processed."""
        return self.n_

    @property
    def mean(self):
        """Return the running mean, or NaN if no values were observed."""
        if self.n_ == 0:
            return np.nan
        return self.mean_

    @property
    def variance(self):
        """Return population variance, or NaN if no values were observed."""
        if self.n_ == 0:
            return np.nan
        return self.M2_ / self.n_

    @property
    def sample_variance(self):
        """Return sample variance with ddof=1, or NaN if count < 2."""
        if self.n_ < 2:
            return np.nan
        return self.M2_ / (self.n_ - 1)

    @property
    def std(self):
        """Return population standard deviation."""
        return float(np.sqrt(self.variance))

    @property
    def sample_std(self):
        """Return sample standard deviation with ddof=1."""
        return float(np.sqrt(self.sample_variance))

    @property
    def min(self):  # noqa: A003
        """Return minimum observed value, or NaN if no values were observed."""
        if self.n_ == 0:
            return np.nan
        return self.min_

    @property
    def max(self):  # noqa: A003
        """Return maximum observed value, or NaN if no values were observed."""
        if self.n_ == 0:
            return np.nan
        return self.max_

    def to_dict(self):
        """
        Return all streaming statistics in a dictionary.

        Returns
        -------
        dict
            Summary containing count, mean, variance, std, min, and max.
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "sample_variance": self.sample_variance,
            "std": self.std,
            "sample_std": self.sample_std,
            "min": self.min,
            "max": self.max,
        }