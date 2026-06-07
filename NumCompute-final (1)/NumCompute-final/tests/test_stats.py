import numpy as np
import pytest

from numcompute.stats import (
    StreamingStats,
    histogram,
    max,
    mean,
    median,
    min,
    quantile,
    std,
)


def test_mean_basic():
    assert mean(np.array([1, 2, 3], dtype=float)) == 2.0


def test_mean_axis_and_nan():
    arr = np.array([[1, 2, np.nan], [4, 5, 6]], dtype=float)
    out = mean(arr, axis=1)
    assert np.allclose(out, np.array([1.5, 5.0]))


def test_mean_axis_0_shape():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    out = mean(arr, axis=0)
    assert out.shape == (3,)
    assert np.allclose(out, np.array([2.5, 3.5, 4.5]))


def test_median_basic():
    assert median(np.array([1, 3, 2], dtype=float)) == 2.0


def test_median_axis():
    arr = np.array([[1, 9, 3], [4, 5, 6]], dtype=float)
    out = median(arr, axis=1)
    assert np.allclose(out, np.array([3.0, 5.0]))


def test_std_matches_numpy():
    arr = np.array([1, 2, 3], dtype=float)
    assert np.isclose(std(arr), np.std(arr))


def test_std_sample_ddof():
    arr = np.array([1, 2, 3, 4], dtype=float)
    assert np.isclose(std(arr, ddof=1), np.std(arr, ddof=1))


def test_min_and_max_ignore_nan():
    arr = np.array([1, np.nan, 3], dtype=float)
    assert min(arr) == 1.0
    assert max(arr) == 3.0


def test_min_and_max_axis():
    arr = np.array([[1, 4, 2], [3, 0, 9]], dtype=float)
    assert np.allclose(min(arr, axis=1), np.array([1.0, 0.0]))
    assert np.allclose(max(arr, axis=0), np.array([3.0, 4.0, 9.0]))


def test_histogram_basic():
    counts, bins = histogram(np.array([1, 2, 3], dtype=float), bins=2)
    assert counts.sum() == 3
    assert bins.shape == (3,)


def test_histogram_ignores_nan():
    counts, _ = histogram(np.array([1, 2, np.nan, 3], dtype=float), bins=2)
    assert counts.sum() == 3


def test_histogram_invalid_bins():
    with pytest.raises(ValueError):
        histogram(np.array([1, 2, 3], dtype=float), bins=0)


def test_histogram_invalid_range():
    with pytest.raises(ValueError):
        histogram(np.array([1, 2, 3], dtype=float), bins=2, range=(5, 1))


def test_quantile_basic():
    arr = np.array([1, 2, 3], dtype=float)
    assert quantile(arr, 0.5) == 2.0


def test_quantile_axis():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    out = quantile(arr, 0.5, axis=1)
    assert np.allclose(out, np.array([2.0, 5.0]))


def test_quantile_multiple_q():
    arr = np.array([1, 2, 3, 4], dtype=float)
    out = quantile(arr, [0.25, 0.5, 0.75])
    expected = np.nanquantile(arr, [0.25, 0.5, 0.75])
    assert np.allclose(out, expected)


def test_quantile_lower_interpolation():
    arr = np.array([1, 2, 3, 4], dtype=float)
    assert quantile(arr, 0.5, interpolation="lower") == 2.0


def test_quantile_invalid_q_raises():
    with pytest.raises(ValueError):
        quantile(np.array([1, 2, 3], dtype=float), 1.5)


def test_quantile_invalid_interpolation_raises():
    with pytest.raises(ValueError):
        quantile(np.array([1, 2, 3], dtype=float), 0.5, interpolation="bad")


def test_invalid_axis_raises():
    with pytest.raises(ValueError):
        mean(np.array([1, 2, 3], dtype=float), axis=2)


def test_non_integer_axis_raises():
    with pytest.raises(TypeError):
        mean(np.array([1, 2, 3], dtype=float), axis=1.2)


def test_scalar_input_raises():
    with pytest.raises(ValueError):
        mean(5)


def test_streaming_stats_mean_variance_and_std():
    data = np.array([1, 2, 3, 4, 5], dtype=float)

    stats = StreamingStats()
    stats.update_many(data)

    assert stats.count == 5
    assert np.isclose(stats.mean, np.mean(data))
    assert np.isclose(stats.variance, np.var(data))
    assert np.isclose(stats.sample_variance, np.var(data, ddof=1))
    assert np.isclose(stats.std, np.std(data))
    assert np.isclose(stats.sample_std, np.std(data, ddof=1))


def test_streaming_stats_update_one_at_a_time():
    stats = StreamingStats()

    stats.update(10)
    stats.update(20)
    stats.update(30)

    assert stats.count == 3
    assert np.isclose(stats.mean, 20.0)


def test_streaming_stats_ignores_nan():
    data = np.array([1, 2, np.nan, 3], dtype=float)

    stats = StreamingStats()
    stats.update_many(data)

    assert stats.count == 3
    assert np.isclose(stats.mean, 2.0)


def test_streaming_stats_min_max():
    data = np.array([5, 1, 9, 3], dtype=float)

    stats = StreamingStats()
    stats.update_many(data)

    assert stats.min == 1.0
    assert stats.max == 9.0


def test_streaming_stats_empty_returns_nan():
    stats = StreamingStats()

    assert stats.count == 0
    assert np.isnan(stats.mean)
    assert np.isnan(stats.variance)
    assert np.isnan(stats.sample_variance)
    assert np.isnan(stats.std)
    assert np.isnan(stats.sample_std)
    assert np.isnan(stats.min)
    assert np.isnan(stats.max)


def test_streaming_stats_to_dict():
    stats = StreamingStats().update_many([1, 2, 3])
    summary = stats.to_dict()

    assert summary["count"] == 3
    assert np.isclose(summary["mean"], 2.0)
    assert "variance" in summary
    assert "sample_std" in summary
