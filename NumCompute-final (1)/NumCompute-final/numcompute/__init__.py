"""Public package exports for NumCompute.

This file exposes the main public API so users can import common tools directly
from the package, for example:

    from numcompute import StandardScaler, Pipeline, accuracy
"""

from .benchmarking import (
    benchmark_function,
    compare_functions,
    format_benchmark_table,
    loop_sum_of_squares,
    vectorized_sum_of_squares,
)
from .io import load_csv
from .metrics import accuracy, confusion_matrix, f1, mse, precision, recall
from .optim import grad, jacobian
from .pipeline import FeatureUnion, Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .rank import percentile, rank
from .sort_search import (
    binary_search,
    lexsort_rows,
    multi_key_sort,
    quickselect,
    sort_array,
    top_k,
    topk,
)
from .stats import (
    StreamingStats,
    histogram,
    max,
    mean,
    median,
    min,
    quantile,
    std,
)
from .utils import (
    check_array,
    cosine_similarity,
    euclidean_distance,
    logsumexp,
    make_batches,
    manhattan_distance,
    pairwise_euclidean,
    relu,
    sigmoid,
    stable_softmax,
    topk_indices,
    topk_values,
)

__version__ = "0.1.0"

__all__ = [
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
]
