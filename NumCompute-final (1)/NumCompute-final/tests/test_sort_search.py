import numpy as np
import pytest

from numcompute.sort_search import (
    binary_search,
    lexsort_rows,
    multi_key_sort,
    quickselect,
    sort_array,
    top_k,
    topk,
)


def test_sort_array_basic_stable_default():
    arr = np.array([3, 1, 2])
    assert np.array_equal(sort_array(arr), np.array([1, 2, 3]))


def test_sort_array_axis():
    arr = np.array([[3, 1, 2], [9, 7, 8]])
    out = sort_array(arr, axis=1)
    assert np.array_equal(out, np.array([[1, 2, 3], [7, 8, 9]]))


def test_lexsort_rows_by_columns():
    arr = np.array([[2, 1], [1, 2], [1, 1]])
    out = lexsort_rows(arr, keys=[0, 1])
    assert np.array_equal(out, np.array([[1, 1], [1, 2], [2, 1]]))


def test_multi_key_sort_alias():
    arr = np.array([[2, 1], [1, 2], [1, 1]])
    out = multi_key_sort(arr, keys=[0, 1])
    assert np.array_equal(out, np.array([[1, 1], [1, 2], [2, 1]]))


def test_lexsort_rows_descending_single_key():
    arr = np.array([[1, 10], [3, 30], [2, 20]])
    out = lexsort_rows(arr, keys=[0], ascending=False)
    assert np.array_equal(out, np.array([[3, 30], [2, 20], [1, 10]]))


def test_lexsort_rows_invalid_input_raises():
    with pytest.raises(ValueError):
        lexsort_rows(np.array([1, 2, 3]), keys=[0])


def test_lexsort_rows_invalid_key_raises():
    with pytest.raises(ValueError):
        lexsort_rows(np.array([[1, 2], [3, 4]]), keys=[2])


def test_topk_largest_indices_default():
    arr = np.array([1, 9, 3, 7, 5])
    idx = topk(arr, 3, largest=True)
    assert np.array_equal(idx, np.array([1, 3, 4]))


def test_topk_largest_values_sorted():
    arr = np.array([1, 9, 3, 7, 5])
    out = topk(arr, 3, largest=True, return_indices=False)
    assert np.array_equal(out, np.array([9, 7, 5]))


def test_top_k_backward_compatible_values_default():
    arr = np.array([1, 9, 3, 7, 5])
    out = top_k(arr, 3, largest=True)
    assert np.array_equal(out, np.array([9, 7, 5]))


def test_topk_smallest_indices():
    arr = np.array([8, 2, 4, 1])
    idx = topk(arr, 2, largest=False, return_indices=True)
    assert np.array_equal(idx, np.array([3, 1]))


def test_topk_unsorted_output_contains_correct_values():
    arr = np.array([1, 9, 3, 7, 5])
    out = topk(arr, 2, largest=True, return_indices=False, sorted_output=False)
    assert set(out.tolist()) == {9, 7}


def test_topk_invalid_k_raises():
    with pytest.raises(ValueError):
        topk(np.array([1, 2, 3]), 0)

    with pytest.raises(ValueError):
        topk(np.array([1, 2, 3]), 4)


def test_topk_non_1d_raises():
    with pytest.raises(ValueError):
        topk(np.array([[1, 2], [3, 4]]), 1)


def test_binary_search_found_and_not_found():
    arr = np.array([1, 3, 5, 7])
    assert binary_search(arr, 5) == (2, True)
    assert binary_search(arr, 4) == (2, False)


def test_binary_search_insert_beginning_and_end():
    arr = np.array([1, 3, 5, 7])
    assert binary_search(arr, 0) == (0, False)
    assert binary_search(arr, 9) == (4, False)


def test_binary_search_first_duplicate():
    arr = np.array([1, 2, 2, 2, 3])
    assert binary_search(arr, 2) == (1, True)


def test_binary_search_invalid_input_raises():
    with pytest.raises(ValueError):
        binary_search(np.array([[1, 2], [3, 4]]), 3)


def test_quickselect_kth_smallest():
    arr = np.array([7, 1, 3, 9, 5])
    assert quickselect(arr, 2) == 5


def test_quickselect_min_and_max():
    arr = np.array([7, 1, 3, 9, 5])
    assert quickselect(arr, 0) == 1
    assert quickselect(arr, 4) == 9


def test_quickselect_with_duplicates():
    arr = np.array([4, 2, 2, 9, 1])
    assert quickselect(arr, 2) == 2


def test_quickselect_does_not_modify_input():
    arr = np.array([7, 1, 3, 9, 5])
    original = arr.copy()
    quickselect(arr, 2)
    assert np.array_equal(arr, original)


def test_quickselect_invalid_k_raises():
    with pytest.raises(ValueError):
        quickselect(np.array([1, 2, 3]), -1)

    with pytest.raises(ValueError):
        quickselect(np.array([1, 2, 3]), 3)
