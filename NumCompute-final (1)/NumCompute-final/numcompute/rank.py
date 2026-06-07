"""Ranking and percentile helpers for NumCompute.

This module provides tie-aware ranking and percentile utilities using NumPy.
Ranks start at 1. NaN values in rank inputs are ignored and returned as NaN
in the output rank array.
"""

from __future__ import annotations

import numpy as np


_ALLOWED_RANK_METHODS = {"average", "dense", "ordinal"}
_ALLOWED_PERCENTILE_METHODS = {"linear", "lower", "higher", "midpoint"}


def _as_1d_float_array(data, name="data") -> np.ndarray:
    """Convert input to a 1D floating NumPy array."""
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    return arr


def rank(data, method="average"):
    """
    Rank 1D data with tie handling.

    Parameters
    ----------
    data : array-like of shape (n,)
        Input values to rank.
    method : {"average", "dense", "ordinal"}, default="average"
        Ranking strategy:
        - "average": tied values receive the average of their ordinal ranks.
        - "dense": tied values receive the same rank, with no gaps.
        - "ordinal": every value receives a unique rank based on stable order.

    Returns
    -------
    np.ndarray
        Float array of ranks starting at 1. NaN input values receive NaN ranks.

    Examples
    --------
    >>> rank([10, 20, 20, 30], method="average")
    array([1. , 2.5, 2.5, 4. ])
    """
    if method not in _ALLOWED_RANK_METHODS:
        raise ValueError(
            "method must be one of {'average', 'dense', 'ordinal'}."
        )

    x = _as_1d_float_array(data)
    ranks = np.full(x.shape, np.nan, dtype=float)

    valid_mask = ~np.isnan(x)

    if not np.any(valid_mask):
        return ranks

    valid_values = x[valid_mask]
    valid_positions = np.flatnonzero(valid_mask)

    order = np.argsort(valid_values, kind="stable")
    sorted_values = valid_values[order]
    sorted_positions = valid_positions[order]

    if method == "ordinal":
        ranks[sorted_positions] = np.arange(1, sorted_values.size + 1, dtype=float)
        return ranks

    unique_values, first_indices, counts = np.unique(
        sorted_values,
        return_index=True,
        return_counts=True,
    )

    if method == "dense":
        dense_ranks = np.arange(1, unique_values.size + 1, dtype=float)
        sorted_ranks = np.repeat(dense_ranks, counts)
    else:
        # Average rank for a tie group occupying positions:
        # first_index + 1 through first_index + count.
        sorted_ranks = np.repeat(first_indices + 1 + (counts - 1) / 2.0, counts)

    ranks[sorted_positions] = sorted_ranks
    return ranks


def percentile(data, q, interpolation="linear"):
    """
    Compute percentile(s) of an array while ignoring NaN values.

    Parameters
    ----------
    data : array-like
        Input data. Values are flattened before percentile calculation.
    q : float or array-like
        Percentile or percentiles in [0, 100].
    interpolation : {"linear", "lower", "higher", "midpoint"}, default="linear"
        Interpolation method used when the percentile lies between two values.

    Returns
    -------
    float or np.ndarray
        Percentile value if q is scalar, otherwise an array of percentile values.

    Raises
    ------
    ValueError
        If q is outside [0, 100] or interpolation is unsupported.
    """
    if interpolation not in _ALLOWED_PERCENTILE_METHODS:
        raise ValueError(
            "interpolation must be one of {'linear', 'lower', 'higher', 'midpoint'}."
        )

    x = np.asarray(data, dtype=float)

    if x.ndim == 0:
        raise ValueError("data must have at least one dimension.")

    q_arr = np.asarray(q, dtype=float)
    scalar_q = q_arr.ndim == 0

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100.")

    result = np.nanpercentile(x, q_arr, method=interpolation)

    if scalar_q:
        return float(np.asarray(result))

    return result
