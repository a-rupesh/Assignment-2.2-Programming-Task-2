"""Streaming benchmark suite for NumCompute Stream.

Compares:
1. Vectorised NumPy ops vs Python loops (sum of squares — from base package)
2. Single StreamingDecisionTree vs StreamingBaggingClassifier vs StreamingRandomForest
   under streaming conditions on a synthetic dataset.

Run with:
    python benchmark/stream_bench.py

Output is a formatted table printed to stdout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numcompute_stream.tree import StreamingDecisionTree
from numcompute_stream.ensemble import StreamingBaggingClassifier, StreamingRandomForest
from numcompute_stream.preprocessing import StreamingStandardScaler


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _timed(func, *args, repeat: int = 3, warmup: int = 1, **kwargs):
    """Return (mean_time, result) for repeated calls."""
    for _ in range(warmup):
        func(*args, **kwargs)
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), result


def _fmt_table(rows, headers):
    widths = [max(len(h), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    hdr = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines = [hdr, sep]
    for r in rows:
        lines.append(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark 1: vectorised vs loop (carry-over from base package)
# ---------------------------------------------------------------------------

def bench_vectorised_vs_loop():
    print("\n=== Benchmark 1: Vectorised vs Python loop (sum of squares) ===")
    rng = np.random.default_rng(0)
    x = rng.random(500_000)

    def loop_sos(arr):
        total = 0.0
        for v in arr:
            total += v * v
        return total

    def vec_sos(arr):
        return float(np.dot(arr, arr))

    t_loop, _ = _timed(loop_sos, x, repeat=3)
    t_vec, _ = _timed(vec_sos, x, repeat=5)
    speedup = t_loop / t_vec if t_vec > 0 else float('inf')

    rows = [
        ['loop_sos', f'{t_loop:.4f}s', '1.00×'],
        ['vec_sos',  f'{t_vec:.4f}s',  f'{speedup:.1f}×'],
    ]
    print(_fmt_table(rows, ['Function', 'Mean time', 'Speedup vs loop']))


# ---------------------------------------------------------------------------
# Benchmark 2: streaming model comparison
# ---------------------------------------------------------------------------

def bench_streaming_models():
    print("\n=== Benchmark 2: Streaming model comparison (synthetic dataset) ===")
    rng = np.random.default_rng(42)
    N_TOTAL = 2000
    N_FEATURES = 10
    CHUNK_SIZE = 100
    CLASSES = np.array([0, 1])

    X = rng.standard_normal((N_TOTAL, N_FEATURES))
    y = (X[:, 0] + X[:, 1] + 0.5 * rng.standard_normal(N_TOTAL) > 0).astype(int)
    chunks = [(X[i:i+CHUNK_SIZE], y[i:i+CHUNK_SIZE])
              for i in range(0, N_TOTAL, CHUNK_SIZE)]

    models = {
        'Decision Tree (1 tree)': StreamingDecisionTree(
            min_samples_split=30, delta=1e-5, random_state=0),
        'Bagging (5 trees)': StreamingBaggingClassifier(
            n_estimators=5, min_samples_split=30, delta=1e-5, random_state=0),
        'Random Forest (5 trees, sqrt features)': StreamingRandomForest(
            n_estimators=5, max_features='sqrt', min_samples_split=30,
            delta=1e-5, random_state=0),
    }

    rows = []
    for name, model in models.items():
        scaler = StreamingStandardScaler()

        t0 = time.perf_counter()
        for Xc, yc in chunks:
            scaler.partial_fit(Xc)
            model.partial_fit(scaler.transform(Xc), yc, classes=CLASSES)
        train_time = time.perf_counter() - t0

        X_scaled = scaler.transform(X)
        t0 = time.perf_counter()
        preds = model.predict(X_scaled)
        pred_time = time.perf_counter() - t0

        acc = np.mean(preds == y)
        rows.append([name, f'{train_time:.3f}s', f'{pred_time:.4f}s', f'{acc:.3f}'])

    print(_fmt_table(rows, ['Model', 'Train time', 'Predict time', 'Accuracy']))


# ---------------------------------------------------------------------------
# Benchmark 3: accuracy vs chunk size
# ---------------------------------------------------------------------------

def bench_chunk_size_effect():
    print("\n=== Benchmark 3: Effect of chunk size on streaming accuracy ===")
    rng = np.random.default_rng(7)
    N_TOTAL = 2000
    N_FEATURES = 8
    X = rng.standard_normal((N_TOTAL, N_FEATURES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    CLASSES = np.array([0, 1])

    rows = []
    for chunk_size in [20, 50, 100, 200, 500]:
        model = StreamingDecisionTree(min_samples_split=max(10, chunk_size // 4),
                                      delta=1e-5, random_state=0)
        scaler = StreamingStandardScaler()
        for i in range(0, N_TOTAL, chunk_size):
            Xc = X[i:i+chunk_size]
            yc = y[i:i+chunk_size]
            scaler.partial_fit(Xc)
            model.partial_fit(scaler.transform(Xc), yc, classes=CLASSES)
        acc = np.mean(model.predict(scaler.transform(X)) == y)
        rows.append([str(chunk_size), str(N_TOTAL // chunk_size), f'{acc:.3f}',
                     str(model.n_leaves_), str(model.depth_)])

    print(_fmt_table(rows, ['Chunk size', 'Num chunks', 'Accuracy',
                             'Num leaves', 'Tree depth']))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("NumCompute Stream — Benchmark Suite")
    print("=" * 50)
    bench_vectorised_vs_loop()
    bench_streaming_models()
    bench_chunk_size_effect()
    print("\nDone.")


if __name__ == "__main__":
    main()
