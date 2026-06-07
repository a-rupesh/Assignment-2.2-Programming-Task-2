import numpy as np

from numcompute.benchmarking import (
    benchmark_function,
    compare_functions,
    format_benchmark_table,
    loop_sum_of_squares,
    vectorized_sum_of_squares,
)

import pytest

def test_benchmark_invalid_func():
    with pytest.raises(TypeError):
        benchmark_function(123, np.array([1, 2, 3]))

def test_benchmark_invalid_repeat():
    with pytest.raises(ValueError):
        benchmark_function(vectorized_sum_of_squares, np.array([1, 2]), repeat=0)

def test_benchmark_invalid_warmup():
    with pytest.raises(ValueError):
        benchmark_function(vectorized_sum_of_squares, np.array([1, 2]), warmup=-1)

def test_compare_functions_mismatch():
    def f(x):
        return x + 1

    def g(x):
        return x + 2

    import pytest
    with pytest.raises(ValueError):
        compare_functions(f, g, np.array([1, 2]))

def test_benchmark_function_returns_expected_keys():
    x = np.arange(1000, dtype=float)
    result = benchmark_function(vectorized_sum_of_squares, x, repeat=3, warmup=1)
    assert "mean_time" in result
    assert "result" in result
    assert result["repeat"] == 3


def test_compare_functions_equivalent_outputs():
    x = np.arange(1000, dtype=float)
    result = compare_functions(vectorized_sum_of_squares, loop_sum_of_squares, x, repeat=2, warmup=1)
    assert result["speedup_vs_slow"] >= 1.0
    assert result["faster_name"] in {
        "vectorized_sum_of_squares",
        "loop_sum_of_squares",
    }


def test_format_benchmark_table_contains_names():
    x = np.arange(10, dtype=float)
    result = benchmark_function(vectorized_sum_of_squares, x, repeat=2, warmup=0)
    table = format_benchmark_table([result])
    assert "vectorized_sum_of_squares" in table
    assert "mean_time" in table
