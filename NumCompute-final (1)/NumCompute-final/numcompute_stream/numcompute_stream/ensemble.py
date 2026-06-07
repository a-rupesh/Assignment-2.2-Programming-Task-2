"""Streaming ensemble methods for NumCompute Stream.

Implements two ensemble strategies on top of StreamingDecisionTree:

* StreamingBaggingClassifier — each chunk is fed to all base trees; each
  tree sees a bootstrap subsample of the chunk.
* StreamingRandomForest — extends bagging with random feature subsets at
  each split node, reducing correlation between trees.

Both classes expose the same ``partial_fit`` / ``predict`` interface as a
single StreamingDecisionTree, making them drop-in replacements in a Pipeline.

Notes
-----
- Only NumPy and plain Python; no scikit-learn.
- Predictions are combined by soft majority vote (averaged probabilities).
"""

from __future__ import annotations

import numpy as np
from .tree import StreamingDecisionTree


def _bootstrap_chunk(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple:
    """Draw a bootstrap sample (with replacement) from a chunk."""
    n = X.shape[0]
    idx = rng.choice(n, size=n, replace=True)
    return X[idx], y[idx]


class StreamingBaggingClassifier:
    """Streaming bagging ensemble of decision trees.

    On each ``partial_fit(X, y)`` call every tree receives an independent
    bootstrap sample drawn from (X, y).  Final predictions are made by
    soft majority vote (average of predicted class probabilities).

    Parameters
    ----------
    n_estimators : int, default=10
        Number of base trees.
    max_depth : int, default=10
        Maximum depth of each tree.
    delta : float, default=1e-7
        Hoeffding confidence for each base tree.
    min_samples_split : int, default=50
        Minimum samples per node before splitting.
    n_candidate_thresholds : int, default=10
        Threshold candidates per feature.
    random_state : int or None, default=None
        Reproducibility seed.

    Attributes
    ----------
    estimators_ : list of StreamingDecisionTree
    classes_ : np.ndarray
    n_classes_ : int
    n_features_in_ : int
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        delta: float = 1e-7,
        min_samples_split: int = 50,
        n_candidate_thresholds: int = 10,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.delta = delta
        self.min_samples_split = min_samples_split
        self.n_candidate_thresholds = n_candidate_thresholds
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self.estimators_: list[StreamingDecisionTree] = []
        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self.n_features_in_: int | None = None

    def _make_tree(self) -> StreamingDecisionTree:
        return StreamingDecisionTree(
            max_depth=self.max_depth,
            delta=self.delta,
            min_samples_split=self.min_samples_split,
            n_candidate_thresholds=self.n_candidate_thresholds,
            random_state=int(self._rng.integers(0, 2**31)),
        )

    def _initialise(self, n_features: int, classes: np.ndarray) -> None:
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.n_features_in_ = n_features
        self.estimators_ = [self._make_tree() for _ in range(self.n_estimators)]

    def partial_fit(self, X, y, classes: np.ndarray | None = None) -> "StreamingBaggingClassifier":
        """Train all trees on bootstrap samples from the chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        classes : array-like or None
            Required on the first call.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if self.classes_ is None:
            if classes is not None:
                known = np.sort(np.asarray(classes))
            else:
                known = np.unique(y)
            self._initialise(X.shape[1], known)

        for tree in self.estimators_:
            X_boot, y_boot = _bootstrap_chunk(X, y, self._rng)
            tree.partial_fit(X_boot, y_boot, classes=self.classes_)

        return self

    def fit(self, X, y, classes: np.ndarray | None = None) -> "StreamingBaggingClassifier":
        """Reset all trees and train from scratch."""
        self.estimators_ = []
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y, classes=classes)

    def predict_proba(self, X) -> np.ndarray:
        """Average predicted probabilities across all trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        if not self.estimators_:
            raise ValueError("Call partial_fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        n_valid = 0
        for tree in self.estimators_:
            if tree.root_ is not None:
                proba_sum += tree.predict_proba(X)
                n_valid += 1
        if n_valid == 0:
            return np.full((X.shape[0], self.n_classes_), 1.0 / self.n_classes_)
        return proba_sum / n_valid

    def predict(self, X) -> np.ndarray:
        """Return majority-vote class predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class StreamingRandomForest(StreamingBaggingClassifier):
    """Streaming Random Forest with random feature subsets.

    Extends StreamingBaggingClassifier by limiting the feature candidates at
    each split node to a random subset of size ``max_features``.  This
    de-correlates the trees without additional changes to the base tree
    implementation — it is achieved by randomly masking features before
    passing a chunk to each tree.

    Parameters
    ----------
    n_estimators : int, default=10
    max_features : int, str, or None, default='sqrt'
        Number of features to consider at each split:
        - int: exact number
        - 'sqrt': int(sqrt(n_features))
        - 'log2': int(log2(n_features))
        - None or 'all': all features (degenerates to bagging)
    max_depth : int, default=10
    delta : float, default=1e-7
    min_samples_split : int, default=50
    n_candidate_thresholds : int, default=10
    random_state : int or None, default=None

    Notes
    -----
    Feature masking is applied per chunk (not per node), which is a practical
    compromise for an online setting where we cannot rebuild nodes on the fly.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_features: int | str | None = 'sqrt',
        max_depth: int = 10,
        delta: float = 1e-7,
        min_samples_split: int = 50,
        n_candidate_thresholds: int = 10,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            delta=delta,
            min_samples_split=min_samples_split,
            n_candidate_thresholds=n_candidate_thresholds,
            random_state=random_state,
        )
        self.max_features = max_features
        self._feature_subsets: list[np.ndarray] = []

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None or mf == 'all':
            return n_features
        if mf == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if mf == 'log2':
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, int):
            return max(1, min(mf, n_features))
        raise ValueError(f"Invalid max_features value: {mf!r}")

    def _initialise(self, n_features: int, classes: np.ndarray) -> None:
        super()._initialise(n_features, classes)
        k = self._resolve_max_features(n_features)
        # Assign a fixed feature subset to each tree (stable across chunks)
        self._feature_subsets = [
            np.sort(self._rng.choice(n_features, size=k, replace=False))
            for _ in range(self.n_estimators)
        ]

    def partial_fit(self, X, y, classes: np.ndarray | None = None) -> "StreamingRandomForest":
        """Train all trees on bootstrapped feature-masked chunks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if self.classes_ is None:
            if classes is not None:
                known = np.sort(np.asarray(classes))
            else:
                known = np.unique(y)
            self._initialise(X.shape[1], known)

        for tree, feat_idx in zip(self.estimators_, self._feature_subsets):
            X_boot, y_boot = _bootstrap_chunk(X[:, feat_idx], y, self._rng)
            tree.partial_fit(X_boot, y_boot, classes=self.classes_)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Average probabilities over all trees, applying each tree's feature mask."""
        if not self.estimators_:
            raise ValueError("Call partial_fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        n_valid = 0
        for tree, feat_idx in zip(self.estimators_, self._feature_subsets):
            if tree.root_ is not None:
                proba_sum += tree.predict_proba(X[:, feat_idx])
                n_valid += 1
        if n_valid == 0:
            return np.full((X.shape[0], self.n_classes_), 1.0 / self.n_classes_)
        return proba_sum / n_valid


# ---------------------------------------------------------------------------
# Spec-required alias
# ---------------------------------------------------------------------------

#: Alias for StreamingRandomForest matching the spec class name.
EnsembleClassifier = StreamingRandomForest

