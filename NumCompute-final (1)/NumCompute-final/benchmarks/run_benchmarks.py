

import numpy as np

from numcompute.benchmarking import (
    compare_functions,
    vectorized_sum_of_squares,
    loop_sum_of_squares,
    format_benchmark_table,
)


def benchmark_sum_of_squares():
    """Benchmark sum of squares implementations."""

    print("\n=== Benchmark: Sum of Squares ===")

    # Large input for meaningful timing
    x = np.random.rand(1_000_000)

    result = compare_functions(
        vectorized_sum_of_squares,
        loop_sum_of_squares,
        x,
        repeat=5,
        warmup=1,
    )

    print(format_benchmark_table([result["first"], result["second"]]))
    print(f"\nSpeedup (vectorised vs loop): {result['speedup_vs_slow']:.2f}x")


def benchmark_top_k():
    """Benchmark top-k selection (vectorised vs loop)."""

    print("\n=== Benchmark: Top-K Selection ===")

    from numcompute.sort_search import top_k

    def vectorized_topk(x):
        return top_k(x, k=10)

    def loop_topk(x):
        # naive loop version
        arr = list(x)
        arr.sort(reverse=True)
        return np.array(arr[:10])

    x = np.random.rand(100_000)

    result = compare_functions(
        vectorized_topk,
        loop_topk,
        x,
        repeat=5,
        warmup=1,
    )

    print(format_benchmark_table([result["first"], result["second"]]))
    print(f"\nSpeedup: {result['speedup_vs_slow']:.2f}x")


def benchmark_distance():
    """Benchmark distance calculation."""

    print("\n=== Benchmark: Distance Computation ===")

    from numcompute.utils import pairwise_euclidean

    def vectorized_dist(X, Y):
        return pairwise_euclidean(X, Y)

    def loop_dist(X, Y):
        result = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                result[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
        return result

    X = np.random.rand(300, 10)
    Y = np.random.rand(300, 10)

    result = compare_functions(
        vectorized_dist,
        loop_dist,
        X, Y,
        repeat=3,
        warmup=1,
    )

    print(format_benchmark_table([result["first"], result["second"]]))
    print(f"\nSpeedup: {result['speedup_vs_slow']:.2f}x")


def main():
    print("NumCompute Benchmark Suite")
    print("=" * 40)

    benchmark_sum_of_squares()
    benchmark_top_k()
    benchmark_distance()

    print("\nDone.")


if __name__ == "__main__":
    main()
