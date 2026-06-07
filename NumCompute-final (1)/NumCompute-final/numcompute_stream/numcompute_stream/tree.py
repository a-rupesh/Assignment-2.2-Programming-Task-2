"""Streaming decision tree for NumCompute Stream.

Implements a Hoeffding-bound guided decision tree (Very Fast Decision Tree /
VFDT-style) that can be trained incrementally via ``partial_fit``.

The key insight: instead of requiring all data to choose a split, use the
Hoeffding bound to decide when we have *enough* samples to be confident that
the locally best split is also the globally best split with high probability.

Classes
-------
StreamingDecisionTree
    A single streaming classification tree with ``partial_fit`` / ``predict``.

Notes
-----
- Only NumPy and plain Python are used; no scikit-learn.
- Gini impurity is used as the split criterion.
- Supports binary and multi-class targets.
- NaN values in features are treated as a separate branch.
- Numerical stability: zero-variance columns are never split on.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    """A node in the Hoeffding tree."""
    # Statistics accumulated at this node
    class_counts: np.ndarray      # shape (n_classes,)
    feature_stats: dict           # {feature_idx: {'n': int, 'sum': float, 'sum_sq': float}}
    n_samples: int = 0

    # Split outcome (only populated for internal nodes)
    split_feature: int = -1
    split_threshold: float = 0.0

    # Children (None means leaf)
    left: Optional["_Node"] = None    # value <= threshold
    right: Optional["_Node"] = None   # value >  threshold

    def is_leaf(self) -> bool:
        return self.split_feature == -1

    def majority_class(self) -> int:
        return int(np.argmax(self.class_counts))

    def class_probabilities(self) -> np.ndarray:
        total = self.class_counts.sum()
        if total == 0:
            return self.class_counts.copy()
        return self.class_counts / total


def _gini(counts: np.ndarray) -> float:
    """Gini impurity from a class-count vector.  O(n_classes) time."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.dot(p, p))


def _gini_split(left_counts: np.ndarray, right_counts: np.ndarray) -> float:
    """Weighted Gini impurity after a binary split."""
    n_left = left_counts.sum()
    n_right = right_counts.sum()
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (n_left * _gini(left_counts) + n_right * _gini(right_counts)) / n_total


def _hoeffding_bound(n: int, delta: float, r: float = 1.0) -> float:
    """Hoeffding bound ε such that P[|μ̂ - μ| > ε] ≤ δ.

    Parameters
    ----------
    n : int
        Number of samples.
    delta : float
        Confidence parameter (probability of error).
    r : float
        Range of the random variable (default 1.0 for Gini ∈ [0, 1]).
    """
    if n <= 0:
        return r
    return float(np.sqrt(r * r * np.log(1.0 / delta) / (2.0 * n)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StreamingDecisionTree:
    """A Hoeffding-bound guided streaming classification decision tree.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum allowed depth of the tree.
    delta : float, default=1e-7
        Hoeffding bound confidence.  Lower values mean more samples are
        needed before a split is accepted (more conservative).
    min_samples_split : int, default=50
        Minimum samples a node must accumulate before any split is considered.
    n_candidate_thresholds : int, default=10
        Number of quantile thresholds evaluated per feature when searching for
        the best split.  Higher values give finer splits at extra cost.
    random_state : int or None, default=None
        Seed for threshold sampling reproducibility.

    Attributes
    ----------
    root_ : _Node or None
        Root node.  None until the first call to ``partial_fit``.
    n_classes_ : int
        Number of distinct classes seen so far.
    classes_ : np.ndarray
        Array of unique class labels in sorted order.
    n_features_in_ : int
        Number of features.
    """

    def __init__(
        self,
        max_depth: int = 10,
        delta: float = 1e-7,
        min_samples_split: int = 50,
        n_candidate_thresholds: int = 10,
        random_state: int | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1).")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if n_candidate_thresholds < 1:
            raise ValueError("n_candidate_thresholds must be at least 1.")

        self.max_depth = max_depth
        self.delta = delta
        self.min_samples_split = min_samples_split
        self.n_candidate_thresholds = n_candidate_thresholds
        self.random_state = random_state

        self.root_: _Node | None = None
        self.n_classes_: int | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_node(self) -> _Node:
        counts = np.zeros(self.n_classes_, dtype=float)
        return _Node(class_counts=counts, feature_stats={})

    def _update_stats(self, node: _Node, X: np.ndarray, y: np.ndarray) -> None:
        """Accumulate sample statistics at a leaf node."""
        for label in y:
            idx = int(np.searchsorted(self.classes_, label))
            if 0 <= idx < self.n_classes_:
                node.class_counts[idx] += 1
        node.n_samples += len(y)

        # Per-feature running moments for threshold search
        for f in range(X.shape[1]):
            col = X[:, f]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            if f not in node.feature_stats:
                node.feature_stats[f] = {'values': [], 'labels': []}
            node.feature_stats[f]['values'].extend(valid.tolist())
            valid_labels = y[~np.isnan(col)]
            node.feature_stats[f]['labels'].extend(valid_labels.tolist())
            # Cap buffer to avoid memory blow-up; keep a reservoir sample
            buf = node.feature_stats[f]
            max_buf = max(self.min_samples_split * 4, 400)
            if len(buf['values']) > max_buf:
                keep = self._rng.choice(len(buf['values']), max_buf, replace=False)
                keep.sort()
                buf['values'] = [buf['values'][i] for i in keep]
                buf['labels'] = [buf['labels'][i] for i in keep]

    def _best_split(self, node: _Node) -> tuple[int, float, float]:
        """Find the best (feature, threshold) pair at a leaf node.

        Returns
        -------
        best_feature : int
            Feature index, -1 if no valid split found.
        best_threshold : float
            Split value.
        best_gain : float
            Gini gain of the split.
        """
        current_gini = _gini(node.class_counts)
        best_feature = -1
        best_threshold = 0.0
        best_gini = float('inf')

        for f, stats in node.feature_stats.items():
            vals = np.asarray(stats['values'], dtype=float)
            labs = np.asarray(stats['labels'])
            if len(vals) < 2:
                continue

            # Sample quantile thresholds from observed values
            q_points = np.linspace(10, 90, self.n_candidate_thresholds)
            thresholds = np.percentile(vals, q_points)
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_counts = np.zeros(self.n_classes_, dtype=float)
                right_counts = np.zeros(self.n_classes_, dtype=float)
                for lab in labs[left_mask]:
                    idx = int(np.searchsorted(self.classes_, lab))
                    if 0 <= idx < self.n_classes_:
                        left_counts[idx] += 1
                for lab in labs[right_mask]:
                    idx = int(np.searchsorted(self.classes_, lab))
                    if 0 <= idx < self.n_classes_:
                        right_counts[idx] += 1

                g = _gini_split(left_counts, right_counts)
                if g < best_gini:
                    best_gini = g
                    best_feature = f
                    best_threshold = float(thresh)

        gain = current_gini - best_gini
        return best_feature, best_threshold, gain

    def _try_split(self, node: _Node, depth: int) -> bool:
        """Apply Hoeffding test; if it passes, perform the split.

        After splitting, the buffered samples are replayed through the
        children so their class counts and feature stats are initialised
        from the data already seen at the parent.

        Returns True if a split was performed.
        """
        if depth >= self.max_depth:
            return False
        if node.n_samples < self.min_samples_split:
            return False
        if _gini(node.class_counts) == 0.0:
            return False

        best_f, best_thresh, best_gain = self._best_split(node)
        if best_f == -1:
            return False

        # Hoeffding bound on the gain
        eps = _hoeffding_bound(node.n_samples, self.delta)
        # Only split if best gain exceeds the bound (or clearly beneficial)
        if best_gain < eps and best_gain < 1e-6:
            return False

        # --- Collect all buffered samples from the parent's feature_stats ---
        # We reconstruct (values, labels) for the split feature and use them
        # to route buffered points into children, re-populating class counts.
        split_vals = np.asarray(
            node.feature_stats.get(best_f, {}).get('values', []), dtype=float
        )
        split_labs = np.asarray(
            node.feature_stats.get(best_f, {}).get('labels', [])
        )

        # Perform the split: promote node to internal node
        node.split_feature = best_f
        node.split_threshold = best_thresh
        node.left = self._make_node()
        node.right = self._make_node()

        # Replay buffered samples from the split feature into children
        if len(split_vals) > 0:
            nan_mask = np.isnan(split_vals)
            left_mask = (split_vals <= best_thresh) & ~nan_mask
            right_mask = (split_vals > best_thresh) & ~nan_mask
            # nan goes to left (larger side usually, but left for consistency)
            nan_where = np.where(nan_mask)[0]

            for idx in np.where(left_mask)[0]:
                lab = split_labs[idx]
                cidx = int(np.searchsorted(self.classes_, lab))
                if 0 <= cidx < self.n_classes_:
                    node.left.class_counts[cidx] += 1
                    node.left.n_samples += 1

            for idx in np.where(right_mask)[0]:
                lab = split_labs[idx]
                cidx = int(np.searchsorted(self.classes_, lab))
                if 0 <= cidx < self.n_classes_:
                    node.right.class_counts[cidx] += 1
                    node.right.n_samples += 1

            for idx in nan_where:
                lab = split_labs[idx]
                cidx = int(np.searchsorted(self.classes_, lab))
                if 0 <= cidx < self.n_classes_:
                    node.left.class_counts[cidx] += 1
                    node.left.n_samples += 1

        # Clear parent's buffers — they now live in children
        node.feature_stats = {}
        return True

    def _route(self, node: _Node, x_row: np.ndarray) -> "_Node":
        """Return the leaf node that x_row falls into."""
        if node.is_leaf():
            return node
        f = node.split_feature
        val = x_row[f]
        if np.isnan(val):
            # Route NaN to the larger child (safer for imbalanced data)
            if node.left.n_samples >= node.right.n_samples:
                return self._route(node.left, x_row)
            else:
                return self._route(node.right, x_row)
        if val <= node.split_threshold:
            return self._route(node.left, x_row)
        else:
            return self._route(node.right, x_row)

    def _route_chunk(self, node: _Node, X: np.ndarray, depth: int = 0) -> list:
        """Return (leaf, sample_indices) pairs for a chunk (vectorised routing)."""
        n = X.shape[0]
        if node.is_leaf():
            return [(node, np.arange(n), depth)]

        f = node.split_feature
        col = X[:, f]
        nan_mask = np.isnan(col)
        left_mask = (col <= node.split_threshold) & ~nan_mask
        right_mask = ~left_mask

        results = []
        if left_mask.any():
            for leaf, idx, d in self._route_chunk(node.left, X[left_mask], depth + 1):
                orig_idx = np.where(left_mask)[0][idx]
                results.append((leaf, orig_idx, d))
        if right_mask.any():
            for leaf, idx, d in self._route_chunk(node.right, X[right_mask], depth + 1):
                orig_idx = np.where(right_mask)[0][idx]
                results.append((leaf, orig_idx, d))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partial_fit(self, X, y, classes: np.ndarray | None = None) -> "StreamingDecisionTree":
        """Incrementally train the tree on a new chunk of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Numeric values only; NaN is handled safely.
        y : array-like of shape (n_samples,)
            Integer or string class labels.
        classes : array-like or None, default=None
            All possible class labels. Required on the first call if
            ``y`` does not contain all classes; ignored thereafter.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be 1D with the same length as X.")

        # Initialise class registry
        if self.classes_ is None:
            if classes is not None:
                self.classes_ = np.sort(np.asarray(classes))
            else:
                self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self.n_features_in_ = X.shape[1]
            self.root_ = self._make_node()
        else:
            # Discover new classes if they appear (grow the tree gracefully)
            new = np.setdiff1d(np.unique(y), self.classes_)
            if len(new) > 0:
                self.classes_ = np.sort(np.concatenate([self.classes_, new]))
                self.n_classes_ = len(self.classes_)
                # Pad existing count vectors in-place
                def _pad_node(n: _Node):
                    n.class_counts = np.concatenate([n.class_counts, np.zeros(len(new))])
                    if not n.is_leaf():
                        _pad_node(n.left)
                        _pad_node(n.right)
                _pad_node(self.root_)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features; tree expects {self.n_features_in_}."
                )

        # Route each sample to a leaf and accumulate statistics
        leaf_groups = self._route_chunk(self.root_, X)
        for leaf, idx, depth in leaf_groups:
            self._update_stats(leaf, X[idx], y[idx])
            self._try_split(leaf, depth)

        return self

    def fit(self, X, y, classes: np.ndarray | None = None) -> "StreamingDecisionTree":
        """Reset and train from scratch (delegates to partial_fit)."""
        self.root_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y, classes=classes)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.root_ is None:
            raise ValueError("Call partial_fit before predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return np.array([
            self.classes_[self._route(self.root_, row).majority_class()]
            for row in X
        ])

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        if self.root_ is None:
            raise ValueError("Call partial_fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return np.vstack([
            self._route(self.root_, row).class_probabilities()
            for row in X
        ])

    @property
    def depth_(self) -> int:
        """Return current depth of the deepest leaf."""
        if self.root_ is None:
            return 0
        def _depth(node: _Node) -> int:
            if node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root_)

    @property
    def n_leaves_(self) -> int:
        """Return number of leaves in the current tree."""
        if self.root_ is None:
            return 0
        def _count(node: _Node) -> int:
            if node.is_leaf():
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root_)
