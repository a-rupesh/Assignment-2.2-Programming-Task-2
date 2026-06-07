import numpy as np

import numcompute as nc


EXPECTED_EXPORTS = {
    "__version__",

    # IO
    "load_csv",

    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "SimpleImputer",
    "OneHotEncoder",

    # Sorting / searching
    "sort_array",
    "lexsort_rows",
    "multi_key_sort",
    "topk",
    "top_k",
    "binary_search",
    "quickselect",

    # Ranking
    "rank",
    "percentile",

    # Statistics
    "mean",
    "median",
    "std",
    "min",
    "max",
    "histogram",
    "quantile",
    "StreamingStats",

    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",

    # Optimisation
    "grad",
    "jacobian",

    # Pipeline
    "Pipeline",
    "FeatureUnion",

    # Utilities
    "check_array",
    "sigmoid",
    "relu",
    "logsumexp",
    "stable_softmax",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_similarity",
    "pairwise_euclidean",
    "topk_indices",
    "topk_values",
    "make_batches",

    # Benchmarking
    "benchmark_function",
    "compare_functions",
    "format_benchmark_table",
    "vectorized_sum_of_squares",
    "loop_sum_of_squares",
}


def test_version_exists():
    assert isinstance(nc.__version__, str)
    assert nc.__version__ == "0.1.0"


def test_all_expected_exports_are_listed():
    missing = EXPECTED_EXPORTS - set(nc.__all__)
    assert not missing, f"Missing from __all__: {sorted(missing)}"


def test_all_listed_exports_exist_on_package():
    missing = [name for name in nc.__all__ if not hasattr(nc, name)]
    assert not missing, f"Listed in __all__ but not exported: {missing}"


def test_core_preprocessing_export_works():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = nc.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0)


def test_sort_search_exports_work():
    values = np.array([10, 30, 20, 50])

    idx = nc.topk(values, k=2)
    vals = nc.top_k(values, k=2)

    assert np.array_equal(idx, np.array([3, 1]))
    assert np.array_equal(vals, np.array([50, 30]))
    assert nc.binary_search(np.array([1, 3, 5]), 3) == (1, True)


def test_streaming_stats_export_works():
    stats = nc.StreamingStats()
    stats.update_many([1, 2, 3, np.nan])

    assert stats.count == 3
    assert np.isclose(stats.mean, 2.0)


def test_utils_exports_work():
    assert np.isclose(nc.euclidean_distance([0, 0], [3, 4]), 5.0)
    assert np.isclose(nc.manhattan_distance([1, 2], [3, 4]), 4.0)
    assert np.isclose(nc.cosine_similarity([1, 0], [1, 0]), 1.0)

    values = np.array([1, 9, 3, 7])
    assert np.array_equal(nc.topk_values(values, 2), np.array([9, 7]))


def test_optim_exports_work():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    out = nc.grad(f, np.array([3.0, 4.0]))
    assert np.allclose(out, np.array([6.0, 8.0]), atol=1e-4)


def test_benchmark_exports_work():
    x = np.array([1.0, 2.0, 3.0])
    assert nc.vectorized_sum_of_squares(x) == 14.0
    assert nc.loop_sum_of_squares(x) == 14.0
