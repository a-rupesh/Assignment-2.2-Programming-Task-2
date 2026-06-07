"""Shared utility functions for NumCompute.

This module contains small reusable helpers used across the toolkit:
- array validation
- numerically stable activation functions
- logsumexp and softmax
- distance calculations
- top-k helpers
- mini-batch generation
"""

from __future__ import annotations

import numpy as np


def check_array(
    arr,
    ensure_1d=False,
    ensure_2d=False,
    dtype=float,
    allow_nan=True,
    allow_inf=True,
    copy=False,
):
    """
    Validate and convert input to a NumPy array.

    Parameters
    ----------
    arr : array-like
        Input data.
    ensure_1d : bool, default=False
        Require a 1D array.
    ensure_2d : bool, default=False
        Require a 2D array.
    dtype : data-type or None, default=float
        Desired dtype. If None, dtype is inferred.
    allow_nan : bool, default=True
        Whether NaN values are allowed.
    allow_inf : bool, default=True
        Whether infinite values are allowed.
    copy : bool, default=False
        Whether to force a copy.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    ValueError
        If shape or value constraints are violated.
    """
    if ensure_1d and ensure_2d:
        raise ValueError("ensure_1d and ensure_2d cannot both be True.")

    out = np.asarray(arr, dtype=dtype)

    if copy:
        out = out.copy()

    if out.ndim == 0:
        raise ValueError("Input must have at least one dimension.")

    if ensure_1d and out.ndim != 1:
        raise ValueError("Expected a 1D array.")

    if ensure_2d and out.ndim != 2:
        raise ValueError("Expected a 2D array.")

    if np.issubdtype(out.dtype, np.number):
        if not allow_nan and np.isnan(out).any():
            raise ValueError("NaN values are not allowed.")

        if not allow_inf and np.isinf(out).any():
            raise ValueError("Infinite values are not allowed.")

    return out


def sigmoid(x):
    """
    Compute the element-wise logistic sigmoid in a numerically stable way.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    np.ndarray
        Sigmoid output with values in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)

    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)

    return out


def relu(x):
    """
    Compute the element-wise rectified linear unit.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    np.ndarray
        max(x, 0) element-wise.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def logsumexp(x, axis=None, keepdims=False):
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : array-like
        Input values.
    axis : int or None, default=None
        Axis over which to reduce.
    keepdims : bool, default=False
        Whether to keep reduced dimensions.

    Returns
    -------
    float or np.ndarray
        Stable log-sum-exp result.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("x must not be empty.")

    xmax = np.max(x, axis=axis, keepdims=True)

    # Avoid invalid -inf - -inf when all values along an axis are -inf.
    safe_xmax = np.where(np.isneginf(xmax), 0.0, xmax)
    shifted = x - safe_xmax
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)

    out = safe_xmax + np.log(summed)
    out = np.where(np.isneginf(xmax), -np.inf, out)

    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)

    if np.ndim(out) == 0:
        return float(out)

    return out


def stable_softmax(x, axis=-1):
    """
    Compute softmax in a numerically stable way.

    Parameters
    ----------
    x : array-like
        Input values.
    axis : int, default=-1
        Axis over which probabilities sum to 1.

    Returns
    -------
    np.ndarray
        Softmax probabilities.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("x must not be empty.")

    xmax = np.max(x, axis=axis, keepdims=True)
    safe_xmax = np.where(np.isneginf(xmax), 0.0, xmax)

    exp_shifted = np.exp(x - safe_xmax)
    denom = np.sum(exp_shifted, axis=axis, keepdims=True)

    return np.divide(
        exp_shifted,
        denom,
        out=np.zeros_like(exp_shifted, dtype=float),
        where=denom != 0,
    )


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two 1D vectors.

    Parameters
    ----------
    a, b : array-like of shape (n_features,)
        Input vectors.

    Returns
    -------
    float
        Euclidean distance.
    """
    a = check_array(a, ensure_1d=True, dtype=float)
    b = check_array(b, ensure_1d=True, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    return float(np.linalg.norm(a - b))


def manhattan_distance(a, b):
    """
    Compute Manhattan / L1 distance between two 1D vectors.
    """
    a = check_array(a, ensure_1d=True, dtype=float)
    b = check_array(b, ensure_1d=True, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    return float(np.sum(np.abs(a - b)))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D vectors.

    Returns 0.0 if either vector has zero norm.
    """
    a = check_array(a, ensure_1d=True, dtype=float)
    b = check_array(b, ensure_1d=True, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def pairwise_euclidean(X, Y=None):
    """
    Compute pairwise Euclidean distances between rows of X and Y.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        First input matrix.
    Y : array-like of shape (n_samples_Y, n_features) or None, default=None
        Second input matrix. If None, distances are computed between rows of X.

    Returns
    -------
    np.ndarray of shape (n_samples_X, n_samples_Y)
        Pairwise Euclidean distance matrix.
    """
    X = check_array(X, ensure_2d=True, dtype=float)
    Y = X if Y is None else check_array(Y, ensure_2d=True, dtype=float)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")

    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T
    sq_dist = X_sq + Y_sq - 2.0 * X @ Y.T
    sq_dist = np.maximum(sq_dist, 0.0)

    return np.sqrt(sq_dist)


def topk_indices(values, k, largest=True):
    """
    Return indices of the top-k largest or smallest values.

    Parameters
    ----------
    values : array-like of shape (n,)
        Input values.
    k : int
        Number of indices to return.
    largest : bool, default=True
        If True, return indices of largest values. Otherwise smallest values.

    Returns
    -------
    np.ndarray
        Indices sorted by selected value.
    """
    values = check_array(values, ensure_1d=True, dtype=None)

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k < 1 or k > values.size:
        raise ValueError("k must satisfy 1 <= k <= len(values).")

    if largest:
        idx = np.argpartition(values, -k)[-k:]
        return idx[np.argsort(values[idx], kind="stable")[::-1]]

    idx = np.argpartition(values, k - 1)[:k]
    return idx[np.argsort(values[idx], kind="stable")]


def topk_values(values, k, largest=True):
    """
    Return the top-k largest or smallest values.
    """
    values = np.asarray(values)
    idx = topk_indices(values, k, largest=largest)
    return values[idx]


def make_batches(X, batch_size, *, drop_last=False, shuffle=False, random_state=None):
    """
    Yield mini-batches from the first axis of an array.

    Parameters
    ----------
    X : array-like
        Input data.
    batch_size : int
        Number of samples per batch.
    drop_last : bool, default=False
        If True, drop the final incomplete batch.
    shuffle : bool, default=False
        If True, shuffle rows before batching.
    random_state : int or None, default=None
        Seed used when shuffle=True.

    Yields
    ------
    np.ndarray
        Consecutive or shuffled mini-batches.
    """
    X = np.asarray(X)

    if X.ndim == 0:
        raise ValueError("X must have at least one dimension.")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    n_samples = X.shape[0]

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)

        if drop_last and (stop - start) < batch_size:
            continue

        yield X[indices[start:stop]]
