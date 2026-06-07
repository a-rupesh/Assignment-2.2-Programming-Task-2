# NumCompute Stream

**NumCompute Stream** extends the base NumCompute package with a streaming,
decision tree–based machine learning framework.  All components support
incremental learning via `.partial_fit()` — no need to hold the full dataset
in memory.

> **Assignment 2.2 — Individual submission**  
> Course: Programming in AI

---

## What's new

| Module | What it adds |
|--------|-------------|
| `numcompute_stream/preprocessing.py` | `StreamingStandardScaler`, `StreamingMinMaxScaler` — incremental preprocessing using Welford's algorithm |
| `numcompute_stream/tree.py` | `StreamingDecisionTree` — Hoeffding-bound Very Fast Decision Tree (VFDT) |
| `numcompute_stream/ensemble.py` | `StreamingBaggingClassifier`, `StreamingRandomForest` |
| `numcompute_stream/visualise.py` | `plot_metrics`, `plot_confusion_matrix`, `plot_roc_curve`, `plot_comparison`, `plot_tree_structure` |

---

## Setup

```bash
# Requirements: Python ≥ 3.9, NumPy, matplotlib
pip install numpy matplotlib pytest

# Install in editable mode (optional)
pip install -e .
```

---

## Quick start

```python
import numpy as np
from numcompute_stream.preprocessing import StreamingStandardScaler
from numcompute_stream.tree import StreamingDecisionTree
from numcompute_stream.ensemble import StreamingRandomForest
from numcompute_stream import visualise

# --- Simulate a streaming scenario ---
rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 8))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

scaler = StreamingStandardScaler()
tree = StreamingDecisionTree(min_samples_split=30, delta=1e-5)
classes = np.array([0, 1])

accuracy_log = []

# Feed data in chunks of 100
for i in range(0, 1000, 100):
    Xc, yc = X[i:i+100], y[i:i+100]
    scaler.partial_fit(Xc)
    tree.partial_fit(scaler.transform(Xc), yc, classes=classes)
    acc = np.mean(tree.predict(scaler.transform(X)) == y)
    accuracy_log.append(acc)

# Visualise
visualise.plot_metrics({'accuracy': accuracy_log})
```

---

## Running tests

```bash
pytest tests/ -v
pytest numcompute_stream/test_stream.py -v

---

## Running benchmarks

```bash
python benchmark/stream_bench.py
```

Sample output:

```
=== Benchmark 1: Vectorised vs Python loop (sum of squares) ===
Function | Mean time | Speedup vs loop
loop_sos | 0.0602s   | 1.00×
vec_sos  | 0.0002s   | 321.6×

=== Benchmark 2: Streaming model comparison ===
Model                                  | Train time | Accuracy
Decision Tree (1 tree)                 | 1.46s      | 0.887
Bagging (5 trees)                      | 5.31s      | 0.901
Random Forest (5 trees, sqrt features) | 1.20s      | 0.715
```

---

## Running the demo notebook

```bash
jupyter notebook demo/stream_demo.ipynb
```

The notebook:
1. Generates a synthetic dataset with missing values and saves it as CSV
2. Loads it back with `np.genfromtxt` (mimicking `io.py`)
3. Splits into 20 chunks of 100 samples each
4. Trains three models incrementally, logging metrics after each chunk
5. Plots accuracy/F1 progression, confusion matrices, and ROC curves

---

## API Reference

### StreamingStandardScaler

```python
scaler = StreamingStandardScaler()
scaler.partial_fit(X_chunk)     # update running mean/std (Welford)
X_scaled = scaler.transform(X)  # z-score: (X - mean) / std
scaler.fit(X)                   # reset + fit in one call
```

**Attributes:** `mean_`, `var_`, `std_`, `n_samples_seen_`, `n_features_in_`

---

### StreamingMinMaxScaler

```python
scaler = StreamingMinMaxScaler(feature_range=(0, 1))
scaler.partial_fit(X_chunk)     # expand running min/max
X_scaled = scaler.transform(X)
```

**Attributes:** `data_min_`, `data_max_`, `n_features_in_`

---

### StreamingDecisionTree

```python
tree = StreamingDecisionTree(
    max_depth=10,           # maximum tree depth
    delta=1e-7,             # Hoeffding confidence (lower = more conservative)
    min_samples_split=50,   # minimum samples before considering a split
    n_candidate_thresholds=10,
    random_state=None,
)
tree.partial_fit(X, y, classes=np.array([0, 1]))
preds = tree.predict(X)
proba = tree.predict_proba(X)   # shape (n_samples, n_classes)
```

**Key properties:** `depth_`, `n_leaves_`, `classes_`, `n_features_in_`

**How it works:**  
Implements the Hoeffding bound (VFDT) algorithm.  A leaf accumulates samples
until the Gini gain of the best split exceeds the Hoeffding bound — the
statistical guarantee that this split is near-optimal with probability
`1 - delta`.  Splits fire lazily; the parent's buffered samples are replayed
into children on split to initialise their class counts.

---

### StreamingBaggingClassifier

```python
bag = StreamingBaggingClassifier(
    n_estimators=10,
    max_depth=10,
    delta=1e-7,
    min_samples_split=50,
    random_state=None,
)
bag.partial_fit(X, y, classes=np.array([0, 1]))
preds = bag.predict(X)          # soft majority vote
proba = bag.predict_proba(X)    # averaged probabilities
```

Each chunk is bootstrap-resampled independently for each tree.

---

### StreamingRandomForest

```python
rf = StreamingRandomForest(
    n_estimators=10,
    max_features='sqrt',   # 'sqrt' | 'log2' | int | None/'all'
    max_depth=10,
    delta=1e-7,
    min_samples_split=50,
    random_state=None,
)
rf.partial_fit(X, y, classes=np.array([0, 1]))
preds = rf.predict(X)
```

Each tree is assigned a fixed random feature subset at initialisation.
This reduces tree correlation compared to plain Bagging.

---

### visualise module

```python
from numcompute_stream import visualise

# Metric progression
visualise.plot_metrics({'accuracy': [...], 'f1': [...]})

# Confusion matrix
visualise.plot_confusion_matrix(cm, class_names=['Neg', 'Pos'], normalize=True)

# ROC curve
visualise.plot_roc_curve(fpr, tpr, auc_score=0.91)

# Model comparison
visualise.plot_comparison({
    'Tree': {'accuracy': [...]},
    'Forest': {'accuracy': [...]},
}, metric_name='accuracy')

# Tree structure (text)
print(visualise.plot_tree_structure(tree, max_depth=4))
```

All functions accept an optional `ax` argument to embed into existing
matplotlib layouts, and a `save_path` to write the figure to disk.

---

## Design decisions

### Hoeffding bound for splits
Rather than the batch approach of scanning all data to find the best split,
the VFDT uses Hoeffding's inequality:

```
ε = sqrt(R² · ln(1/δ) / (2n))
```

A split fires when the Gini gain of the locally-best feature exceeds ε,
guaranteeing (with probability 1−δ) that the same split would have been
chosen given infinite data.  This is the key to constant memory usage per
node.

### NaN handling
- Scalers: NaN is ignored during `partial_fit` (Welford update skips NaN).
- Scalers: NaN is preserved through `transform`.
- Tree routing: NaN in the split feature is routed to the larger child.
- Tree statistics: NaN rows are excluded from threshold evaluation.

### Numerical stability
- Gini impurity uses `np.dot(p, p)` (avoids repeated power).
- Scaler: constant columns get `std = 1.0` to avoid division by zero.
- MinMaxScaler: zero-range columns get `range = 1.0`.

---

## Package structure

```
numcompute_stream/
├── numcompute_stream/
│   ├── __init__.py
│   ├── preprocessing.py   # StreamingStandardScaler, StreamingMinMaxScaler
│   ├── tree.py            # StreamingDecisionTree (VFDT)
│   ├── ensemble.py        # StreamingBaggingClassifier, StreamingRandomForest
│   └── visualise.py       # matplotlib plotting functions
├── tests/
│   └── test_stream.py     # 56 unit tests
├── demo/
│   └── stream_demo.ipynb  # end-to-end Jupyter demo
├── benchmark/
│   └── stream_bench.py    # vectorised vs loop + model comparison
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | all numerical operations |
| `matplotlib` | visualise module |
| `pytest` | unit testing |

No scikit-learn, no pandas, no PyTorch.
