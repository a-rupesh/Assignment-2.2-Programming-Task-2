"""Sorting, searching, partial sorting, and selection helpers for NumCompute.

This module implements:
- stable sorting wrappers around NumPy
- multi-key row sorting
- top-k selection using np.argpartition
- educational quickselect
- binary search returning insertion index and existence flag
"""

from __future__ import annotations

import numpy as np


def sort_array(arr, axis=-1, stable=True):
    """
    Sort an array along a given axis.

    Parameters
    ----------
    arr : array-like
        Input array.
    axis : int, default=-1
        Axis along which to sort.
    stable : bool, default=True
        If True, uses NumPy's stable sorting algorithm.

    Returns
    -------
    np.ndarray
        Sorted copy of the input array.
    """
    arr = np.asarray(arr)
    kind = "stable" if stable else "quicksort"
    return np.sort(arr, axis=axis, kind=kind)


def lexsort_rows(arr, keys=None, ascending=True):
    """
    Sort rows of a 2D array using one or more column keys.

    Parameters
    ----------
    arr : array-like of shape (n_rows, n_cols)
        Input 2D array.
    keys : sequence[int] or None, default=None
        Column indices in priority order. If None, all columns are used
        from left to right.
    ascending : bool or sequence[bool], default=True
        Sort direction. A single bool applies to all keys. A sequence gives
        one direction per key.

    Returns
    -------
    np.ndarray
        Sorted rows.

    Examples
    --------
    >>> lexsort_rows([[2, 1], [1, 2], [1, 1]], keys=[0, 1])
    array([[1, 1],
           [1, 2],
           [2, 1]])
    """
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError("arr must be a 2D array.")

    n_cols = arr.shape[1]

    if keys is None:
        keys = list(range(n_cols))

    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index.")

    keys = [int(k) for k in keys]
    normalized_keys = []

    for key in keys:
        if key < -n_cols or key >= n_cols:
            raise ValueError("key index out of bounds.")
        normalized_keys.append((key + n_cols) % n_cols)

    if isinstance(ascending, bool):
        ascending_flags = [ascending] * len(normalized_keys)
    else:
        ascending_flags = list(ascending)
        if len(ascending_flags) != len(normalized_keys):
            raise ValueError("ascending must be a bool or match the length of keys.")
        if not all(isinstance(flag, bool) for flag in ascending_flags):
            raise TypeError("ascending entries must be booleans.")

    order = np.arange(arr.shape[0])

    # Stable multi-key sort: apply lower-priority keys first.
    for key, asc in reversed(list(zip(normalized_keys, ascending_flags))):
        values = arr[order, key]
        key_order = np.argsort(values, kind="stable")
        if not asc:
            key_order = key_order[::-1]
        order = order[key_order]

    return arr[order]


def multi_key_sort(arr, keys=None, ascending=True):
    """
    Alias for lexsort_rows() with a clearer assignment-style name.
    """
    return lexsort_rows(arr, keys=keys, ascending=ascending)


def topk(values, k, largest=True, return_indices=True, sorted_output=True):
    """
    Select the top-k largest or smallest values using np.argpartition.

    Parameters
    ----------
    values : array-like of shape (n,)
        Input values.
    k : int
        Number of elements to select. Must satisfy 1 <= k <= len(values).
    largest : bool, default=True
        If True, select the largest k values. Otherwise select the smallest k.
    return_indices : bool, default=True
        If True, return indices into the original array. If False, return values.
    sorted_output : bool, default=True
        If True, selected items are sorted from largest-to-smallest or
        smallest-to-largest.

    Returns
    -------
    np.ndarray
        Selected indices or selected values.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k < 1 or k > values.size:
        raise ValueError("k must satisfy 1 <= k <= len(values).")

    if largest:
        selected = np.argpartition(values, -k)[-k:]
        if sorted_output:
            selected = selected[np.argsort(values[selected], kind="stable")[::-1]]
    else:
        selected = np.argpartition(values, k - 1)[:k]
        if sorted_output:
            selected = selected[np.argsort(values[selected], kind="stable")]

    return selected if return_indices else values[selected]


def top_k(values, k, largest=True, return_indices=False, sorted_output=True):
    """
    Backward-compatible alias for topk().

    The assignment API is topk(..., return_indices=True). This alias preserves
    earlier project behavior where top_k returned values by default.
    """
    return topk(
        values,
        k,
        largest=largest,
        return_indices=return_indices,
        sorted_output=sorted_output,
    )


def quickselect(arr, k):
    """
    Return the k-th smallest value using an educational quickselect algorithm.

    Parameters
    ----------
    arr : array-like of shape (n,)
        Input values.
    k : int
        Zero-based order statistic. k=0 returns the minimum.

    Returns
    -------
    scalar
        The k-th smallest value.

    Notes
    -----
    This implementation copies the input and does not mutate the caller's array.
    Average time complexity is O(n), worst-case O(n^2).
    """
    values = np.asarray(arr)

    if values.ndim != 1:
        raise ValueError("arr must be a 1D array.")

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k < 0 or k >= values.size:
        raise ValueError("k out of bounds.")

    work = values.copy()

    left = 0
    right = work.size - 1

    while True:
        if left == right:
            return work[left]

        pivot_index = (left + right) // 2
        pivot_index = _partition(work, left, right, pivot_index)

        if k == pivot_index:
            return work[k]
        if k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


def _partition(values, left, right, pivot_index):
    """Partition helper used by quickselect."""
    pivot_value = values[pivot_index]
    values[pivot_index], values[right] = values[right], values[pivot_index]

    store_index = left

    for i in range(left, right):
        if values[i] < pivot_value:
            values[store_index], values[i] = values[i], values[store_index]
            store_index += 1

    values[right], values[store_index] = values[store_index], values[right]
    return store_index


def binary_search(sorted_array, x):
    """
    Binary search over a sorted 1D array.

    Parameters
    ----------
    sorted_array : array-like of shape (n,)
        Sorted input array.
    x : scalar
        Value to search for.

    Returns
    -------
    tuple[int, bool]
        (insertion_index, exists). If exists is False, insertion_index is the
        location where x should be inserted to keep the array sorted.
    """
    arr = np.asarray(sorted_array)

    if arr.ndim != 1:
        raise ValueError("sorted_array must be a 1D array.")

    left = 0
    right = arr.size

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid

    exists = left < arr.size and arr[left] == x
    return int(left), bool(exists)
