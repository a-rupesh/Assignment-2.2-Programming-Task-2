"""
NumCompute Demo Script

Run from the project root:
    python demo/demo.py

This demonstrates:
- CSV loading
- preprocessing
- sorting/searching/top-k/quickselect
- ranking and percentiles
- descriptive and streaming statistics
- metrics including ROC/AUC
- optimisation
- pipeline
- utility helpers
- benchmarking
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


# Make imports work when running: python demo/demo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numcompute as nc
from numcompute.benchmarking import (
    compare_functions,
    format_benchmark_table,
    loop_sum_of_squares,
    vectorized_sum_of_squares,
)
from numcompute.io import load_csv
from numcompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_curve,
)
from numcompute.optim import grad, jacobian
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute.rank import percentile, rank
from numcompute.sort_search import (
    binary_search,
    lexsort_rows,
    multi_key_sort,
    quickselect,
    sort_array,
    top_k,
    topk,
)
from numcompute.stats import (
    StreamingStats,
    histogram,
    max as stats_max,
    mean,
    median,
    min as stats_min,
    quantile,
    std,
)
from numcompute.utils import (
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


def print_title(title: str) -> None:
    """Print a formatted section title."""
    print(f"\n=== {title} ===")


def demo_io() -> np.ndarray:
    """Demonstrate CSV loading with missing values."""
    print_title("CSV Loading")

    csv_text = "1,10,0\n2,,0\n3,30,1\n4,40,1"

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        f.write(csv_text)
        path = f.name

    data = load_csv(path, delimiter=",")
    print("Loaded CSV data:")
    print(data)

    return data


def demo_preprocessing(data: np.ndarray) -> np.ndarray:
    """Demonstrate imputation, scaling, and one-hot encoding."""
    print_title("Preprocessing")

    X_num = data[:, :2]

    imputer = SimpleImputer(strategy="mean", fill_value=0)
    X_imputed = imputer.fit_transform(X_num)

    standard_scaler = StandardScaler()
    X_standard = standard_scaler.fit_transform(X_imputed)

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_minmax = minmax_scaler.fit_transform(X_imputed)

    print("Original numeric data:")
    print(X_num)

    print("\nImputed data:")
    print(X_imputed)

    print("\nStandard scaled data:")
    print(X_standard)

    print("\nMinMax scaled data:")
    print(X_minmax)

    X_cat = np.array(
        [
            ["red", "small"],
            ["blue", "large"],
            ["red", "large"],
            ["green", "small"],
        ],
        dtype=object,
    )

    encoder = OneHotEncoder(handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X_cat)

    print("\nCategories learned:")
    print(encoder.categories_)

    print("\nOne-hot encoded data:")
    print(X_encoded)

    print("\nUnknown category with handle_unknown='ignore':")
    print(encoder.transform(np.array([["purple", "small"]], dtype=object)))

    return X_num


def demo_sort_search_rank() -> None:
    """Demonstrate sort/search, top-k, quickselect, ranking, and percentiles."""
    print_title("Sort / Search / Rank")

    values = np.array([10, 30, 20, 20, 50, 40])

    print("Original values:", values)
    print("Stable sort:", sort_array(values))

    print("\nTop-3 largest indices using topk:")
    print(topk(values, k=3, largest=True, return_indices=True))

    print("\nTop-3 largest values using topk:")
    print(topk(values, k=3, largest=True, return_indices=False))

    print("\nBackward-compatible top_k values:")
    print(top_k(values, k=3, largest=True))

    print("\nQuickselect k=2 (third smallest value):")
    print(quickselect(values, 2))

    sorted_values = sort_array(values)

    print("\nBinary search for 30:")
    print(binary_search(sorted_values, 30))

    print("\nBinary search for missing value 25:")
    print(binary_search(sorted_values, 25))

    rows = np.array(
        [
            [2, 1],
            [1, 2],
            [1, 1],
            [2, 0],
        ]
    )

    print("\nRows:")
    print(rows)

    print("\nLexsort rows by columns [0, 1]:")
    print(lexsort_rows(rows, keys=[0, 1]))

    print("\nMulti-key sort alias:")
    print(multi_key_sort(rows, keys=[0, 1]))

    rank_values = np.array([10, 20, 20, 30, np.nan])

    print("\nRank values:", rank_values)
    print("Average ranks:", rank(rank_values, method="average"))
    print("Dense ranks:", rank(rank_values, method="dense"))
    print("Ordinal ranks:", rank(rank_values, method="ordinal"))

    clean_values = np.array([1, 2, 3, 4])

    print("\nPercentile 50 linear:", percentile(clean_values, 50, interpolation="linear"))
    print("Percentile 50 lower:", percentile(clean_values, 50, interpolation="lower"))
    print("Percentile 50 higher:", percentile(clean_values, 50, interpolation="higher"))
    print("Percentile 50 midpoint:", percentile(clean_values, 50, interpolation="midpoint"))


def demo_stats() -> None:
    """Demonstrate descriptive and streaming statistics."""
    print_title("Statistics")

    values = np.array([1, 2, np.nan, 3, 4, 5], dtype=float)

    print("Values:", values)
    print("Mean:", mean(values))
    print("Median:", median(values))
    print("Standard deviation:", std(values))
    print("Min:", stats_min(values))
    print("Max:", stats_max(values))
    print("Quantile 0.5:", quantile(values, 0.5))

    counts, bins = histogram(values, bins=3)
    print("\nHistogram counts:", counts)
    print("Histogram bins:", bins)

    matrix = np.array([[1, 2, np.nan], [4, 5, 6]], dtype=float)
    print("\nAxis-wise mean over rows:")
    print(mean(matrix, axis=1))

    stream = StreamingStats()
    stream.update_many([1, 2, 3, np.nan, 4, 5])

    print("\nStreamingStats summary:")
    print(stream.to_dict())


def demo_metrics() -> None:
    """Demonstrate classification, regression, ROC, and AUC metrics."""
    print_title("Metrics")

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1(y_true, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    y_true_reg = np.array([1.0, 2.0, 3.0])
    y_pred_reg = np.array([1.2, 1.9, 2.8])

    print("\nMSE:", mse(y_true_reg, y_pred_reg))

    y_true_roc = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, thresholds = roc_curve(y_true_roc, y_scores)
    roc_auc = auc(fpr, tpr)

    print("\nROC FPR:", fpr)
    print("ROC TPR:", tpr)
    print("ROC thresholds:", thresholds)
    print("AUC:", roc_auc)


def demo_optim() -> None:
    """Demonstrate gradient and Jacobian estimation."""
    print_title("Optimisation")

    def f_scalar(x):
        return x[0] ** 2 + x[1] ** 2

    def f_vector(x):
        return np.array(
            [
                x[0] + x[1],
                x[0] * x[1],
            ]
        )

    x = np.array([3.0, 4.0])

    print("Gradient:", grad(f_scalar, x))
    print("Jacobian:")
    print(jacobian(f_vector, x))


def demo_pipeline(X_num: np.ndarray) -> None:
    """Demonstrate a preprocessing + model pipeline."""
    print_title("Pipeline")

    class DummyModel:
        def fit(self, X, y=None):
            self.threshold_ = np.mean(X[:, 0])
            return self

        def predict(self, X):
            return (X[:, 0] > self.threshold_).astype(int)

    y = np.array([0, 0, 1, 1])

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", DummyModel()),
        ]
    )

    pipe.fit(X_num, y)
    preds = pipe.predict(X_num)

    print("Pipeline predictions:", preds)


def demo_utils() -> None:
    """Demonstrate utility helpers."""
    print_title("Utilities")

    a = np.array([0, 0])
    b = np.array([3, 4])

    print("Euclidean distance:", euclidean_distance(a, b))
    print("Manhattan distance:", manhattan_distance(a, b))
    print("Cosine similarity:", cosine_similarity([1, 0], [1, 0]))

    scores = np.array([1.0, 2.0, 3.0])

    print("\nSigmoid:", sigmoid(scores))
    print("ReLU:", relu(np.array([-1.0, 0.0, 2.0])))
    print("LogSumExp:", logsumexp(scores))
    print("Stable softmax:", stable_softmax(scores))

    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Y = np.array([[0.0, 1.0]])

    print("\nPairwise Euclidean:")
    print(pairwise_euclidean(X, Y))

    values = np.array([1, 9, 3, 7])
    print("\nTop-k indices:", topk_indices(values, 2))
    print("Top-k values:", topk_values(values, 2))

    print("\nBatches:")
    for batch in make_batches(np.arange(10), batch_size=4):
        print(batch)


def demo_benchmark() -> None:
    """Demonstrate benchmarking vectorised vs loop implementations."""
    print_title("Benchmark")

    rng = np.random.default_rng(42)
    x = rng.random(100_000)

    result = compare_functions(
        vectorized_sum_of_squares,
        loop_sum_of_squares,
        x,
        repeat=5,
        warmup=1,
    )

    print(format_benchmark_table([result["first"], result["second"]]))
    print("Speedup:", f"{result['speedup_vs_slow']:.2f}x")


def main() -> None:
    """Run the full NumCompute demo."""
    print("NumCompute Demo")
    print("=" * 40)
    print("NumCompute version:", nc.__version__)

    data = demo_io()
    X_num = demo_preprocessing(data)

    demo_sort_search_rank()
    demo_stats()
    demo_metrics()
    demo_optim()
    demo_pipeline(X_num)
    demo_utils()
    demo_benchmark()

    print("\nDone.")


if __name__ == "__main__":
    main()
