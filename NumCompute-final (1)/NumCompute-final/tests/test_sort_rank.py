import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.abspath("numcompute"))

from sort_search import sort_array, top_k, binary_search, quickselect
from rank import rank, percentile


# -------------------------
# sort_search.py tests
# -------------------------

def test_sort_array_1d():
    arr = np.array([5, 2, 4, 1, 3])
    result = sort_array(arr)
    expected = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(result, expected)


def test_sort_array_2d_axis_0():
    arr = np.array([[3, 1], [1, 4], [2, 0]])
    result = sort_array(arr, axis=0)
    expected = np.array([[1, 0], [2, 1], [3, 4]])
    assert np.array_equal(result, expected)


def test_sort_array_2d_axis_1():
    arr = np.array([[3, 1], [4, 2]])
    result = sort_array(arr, axis=1)
    expected = np.array([[1, 3], [2, 4]])
    assert np.array_equal(result, expected)


def test_top_k_length():
    arr = np.array([1, 5, 3, 2, 4])
    result = top_k(arr, 2)
    assert len(result) == 2


def test_top_k_contains_largest_values():
    arr = np.array([1, 5, 3, 2, 4])
    result = top_k(arr, 2)
    assert set(result.tolist()) == {4, 5}


def test_top_k_with_duplicates():
    arr = np.array([5, 1, 5, 2, 3])
    result = top_k(arr, 2)
    assert sorted(result.tolist()) == [5, 5]


def test_top_k_k_equals_length():
    arr = np.array([7, 2, 9])
    result = top_k(arr, 3)
    assert sorted(result.tolist()) == [2, 7, 9]


def test_binary_search_found_middle():
    arr = np.array([1, 2, 3, 4, 5])
    assert binary_search(arr, 3) == (2, True)

def test_binary_search_found_first():
    arr = np.array([1, 2, 3, 4, 5])
    assert binary_search(arr, 1) == (0, True)

def test_binary_search_found_last():
    arr = np.array([1, 2, 3, 4, 5])
    assert binary_search(arr, 5) == (4, True)

def test_binary_search_not_found():
    arr = np.array([1, 2, 3, 4, 5])
    assert binary_search(arr, 10) == (5, False)


def test_quickselect_smallest():
    arr = np.array([7, 2, 9, 1, 5])
    assert quickselect(arr, 0) == 1


def test_quickselect_middle():
    arr = np.array([7, 2, 9, 1, 5])
    assert quickselect(arr, 2) == 5


def test_quickselect_largest():
    arr = np.array([7, 2, 9, 1, 5])
    assert quickselect(arr, 4) == 9


def test_quickselect_with_duplicates():
    arr = np.array([4, 2, 2, 8, 5])
    assert quickselect(arr, 1) == 2


def test_quickselect_k_out_of_bounds_low():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        quickselect(arr, -1)


def test_quickselect_k_out_of_bounds_high():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        quickselect(arr, 3)


# -------------------------
# rank.py tests
# -------------------------

def test_rank_ordinal():
    arr = np.array([30, 10, 20])
    result = rank(arr, method="ordinal")
    expected = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(result, expected)


def test_rank_dense_with_ties():
    arr = np.array([10, 20, 20, 30])
    result = rank(arr, method="dense")
    expected = np.array([1.0, 2.0, 2.0, 3.0])
    assert np.array_equal(result, expected)


def test_rank_average_with_ties():
    arr = np.array([10, 20, 20, 30])
    result = rank(arr, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])
    assert np.allclose(result, expected)


def test_rank_all_equal_average():
    arr = np.array([5, 5, 5])
    result = rank(arr, method="average")
    expected = np.array([2.0, 2.0, 2.0])
    assert np.allclose(result, expected)


def test_rank_invalid_method():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        rank(arr, method="random")


def test_percentile_50():
    arr = np.array([1, 2, 3, 4, 5])
    assert percentile(arr, 50) == 3


def test_percentile_0():
    arr = np.array([1, 2, 3, 4, 5])
    assert percentile(arr, 0) == 1


def test_percentile_100():
    arr = np.array([1, 2, 3, 4, 5])
    assert percentile(arr, 100) == 5


def test_percentile_with_duplicates():
    arr = np.array([1, 2, 2, 2, 5])
    assert percentile(arr, 50) == 2
