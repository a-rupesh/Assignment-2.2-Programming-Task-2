import numpy as np
import pytest

from numcompute.rank import percentile, rank


def test_rank_average_handles_ties():
    x = np.array([10, 20, 20, 30])
    out = rank(x, method="average")
    assert np.allclose(out, np.array([1.0, 2.5, 2.5, 4.0]))


def test_rank_dense_handles_ties_without_gaps():
    x = np.array([10, 20, 20, 30])
    out = rank(x, method="dense")
    assert np.allclose(out, np.array([1.0, 2.0, 2.0, 3.0]))


def test_rank_ordinal_uses_stable_order():
    x = np.array([20, 10, 20])
    out = rank(x, method="ordinal")
    assert np.allclose(out, np.array([2.0, 1.0, 3.0]))


def test_rank_unsorted_negative_values():
    x = np.array([5, -1, 5, 2])
    out = rank(x, method="average")
    assert np.allclose(out, np.array([3.5, 1.0, 3.5, 2.0]))


def test_rank_all_equal_average():
    x = np.array([7, 7, 7])
    out = rank(x, method="average")
    assert np.allclose(out, np.array([2.0, 2.0, 2.0]))


def test_rank_all_equal_dense():
    x = np.array([7, 7, 7])
    out = rank(x, method="dense")
    assert np.allclose(out, np.array([1.0, 1.0, 1.0]))


def test_rank_empty_array_returns_empty():
    out = rank(np.array([]), method="average")
    assert out.shape == (0,)


def test_rank_nan_values_receive_nan_rank():
    x = np.array([3.0, np.nan, 1.0])
    out = rank(x, method="average")
    assert np.allclose(out[[0, 2]], np.array([2.0, 1.0]))
    assert np.isnan(out[1])


def test_rank_all_nan_returns_all_nan():
    out = rank(np.array([np.nan, np.nan]), method="dense")
    assert np.isnan(out).all()


def test_rank_invalid_method_raises():
    with pytest.raises(ValueError):
        rank(np.array([1, 2, 3]), method="bad")


def test_rank_non_1d_raises():
    with pytest.raises(ValueError):
        rank(np.array([[1, 2], [3, 4]]), method="average")


def test_percentile_basic_linear():
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50) == 2.5


def test_percentile_lower():
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50, interpolation="lower") == 2.0


def test_percentile_higher():
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50, interpolation="higher") == 3.0


def test_percentile_midpoint():
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50, interpolation="midpoint") == 2.5


def test_percentile_multiple_q():
    x = np.array([1, 2, 3, 4])
    out = percentile(x, [25, 50, 75])
    expected = np.nanpercentile(x, [25, 50, 75])
    assert np.allclose(out, expected)


def test_percentile_ignores_nan():
    x = np.array([1, 2, np.nan, 3])
    assert percentile(x, 50) == 2.0


def test_percentile_multidimensional_flattens():
    x = np.array([[1, 2], [3, 4]])
    assert percentile(x, 50) == 2.5


def test_percentile_invalid_q_raises():
    with pytest.raises(ValueError):
        percentile(np.array([1, 2, 3]), 101)


def test_percentile_invalid_interpolation_raises():
    with pytest.raises(ValueError):
        percentile(np.array([1, 2, 3]), 50, interpolation="nearest")


def test_percentile_scalar_input_raises():
    with pytest.raises(ValueError):
        percentile(5, 50)
