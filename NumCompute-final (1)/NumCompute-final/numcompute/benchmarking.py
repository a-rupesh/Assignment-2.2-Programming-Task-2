"""Simple reproducible micro-benchmark helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def benchmark_function(func, *args, repeat=5, warmup=1, **kwargs):
    """
    Benchmark a callable over repeated runs.

    Returns a dictionary with timing summary statistics.
    """
    if not callable(func):
        raise TypeError("func must be callable.")
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError("repeat must be a positive integer.")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")

    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    last_result = None
    for _ in range(repeat):
        start = time.perf_counter()
        last_result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    times = np.asarray(times, dtype=float)
    return {
        "name": getattr(func, "__name__", "callable"),
        "repeat": repeat,
        "warmup": warmup,
        "times": times,
        "min_time": float(times.min()),
        "max_time": float(times.max()),
        "mean_time": float(times.mean()),
        "median_time": float(np.median(times)),
        "std_time": float(times.std()),
        "result": last_result,
    }


def compare_functions(func_a, func_b, *args, repeat=5, warmup=1, check_equal=True, **kwargs):
    """
    Benchmark and compare two functions on the same inputs.
    """
    first = benchmark_function(func_a, *args, repeat=repeat, warmup=warmup, **kwargs)
    second = benchmark_function(func_b, *args, repeat=repeat, warmup=warmup, **kwargs)

    if check_equal:
        a_res = np.asarray(first["result"])
        b_res = np.asarray(second["result"])
        if not np.allclose(a_res, b_res, equal_nan=True):
            raise ValueError("Function outputs are not numerically equivalent.")

    slower = max(first["mean_time"], second["mean_time"])
    faster = min(first["mean_time"], second["mean_time"])
    speedup_vs_slow = np.inf if faster == 0 else slower / faster

    return {
        "first": first,
        "second": second,
        "speedup_vs_slow": float(speedup_vs_slow),
        "faster_name": first["name"] if first["mean_time"] <= second["mean_time"] else second["name"],
    }


def format_benchmark_table(results):
    """
    Format benchmark result dictionaries into a plain-text table.
    """
    headers = ["name", "mean_time", "median_time", "min_time", "max_time", "std_time"]
    rows = []
    for result in results:
        rows.append([
            str(result["name"]),
            f"{result['mean_time']:.6e}",
            f"{result['median_time']:.6e}",
            f"{result['min_time']:.6e}",
            f"{result['max_time']:.6e}",
            f"{result['std_time']:.6e}",
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_lines = [" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def vectorized_sum_of_squares(x):
    """Reference vectorized implementation."""
    x = np.asarray(x, dtype=float)
    return np.sum(x * x)


def loop_sum_of_squares(x):
    """Reference Python-loop implementation."""
    total = 0.0
    for value in np.asarray(x, dtype=float):
        total += value * value
    return total
