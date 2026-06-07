"""Unit tests for NumCompute Stream.

Run with:
    pytest tests/ -v

Covers:
- StreamingStandardScaler (standard + edge cases)
- StreamingMinMaxScaler (standard + edge cases)
- StreamingDecisionTree (standard + edge cases + streaming conditions)
- StreamingBaggingClassifier (standard + edge cases)
- StreamingRandomForest (standard + edge cases)
- visualise module (smoke tests for plotting functions)
- Integration: pipeline of scaler + tree

Total: 35 tests
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Allow running tests from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numcompute_stream.preprocessing import StreamingStandardScaler, StreamingMinMaxScaler
from numcompute_stream.tree import StreamingDecisionTree, _gini, _hoeffding_bound
from numcompute_stream.ensemble import StreamingBaggingClassifier, StreamingRandomForest
from numcompute_stream import visualise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_dataset():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 4))
    y = np.digitize(X[:, 0], [-0.5, 0.5])  # classes 0, 1, 2
    return X, y


# ---------------------------------------------------------------------------
# StreamingStandardScaler
# ---------------------------------------------------------------------------

class TestStreamingStandardScaler:

    def test_single_chunk_matches_numpy(self):
        """After one partial_fit, mean/std should match numpy."""
        X = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        scaler = StreamingStandardScaler()
        scaler.partial_fit(X)
        np.testing.assert_allclose(scaler.mean_, np.mean(X, axis=0), rtol=1e-6)

    def test_two_chunks_mean_consistency(self):
        """Running mean after two chunks should equal mean of all data."""
        X1 = np.array([[1., 2.], [3., 4.]])
        X2 = np.array([[5., 6.], [7., 8.]])
        scaler = StreamingStandardScaler()
        scaler.partial_fit(X1).partial_fit(X2)
        expected_mean = np.mean(np.vstack([X1, X2]), axis=0)
        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-6)

    def test_transform_produces_zero_mean(self):
        """Transformed data from the same chunk should have near-zero mean."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        scaler = StreamingStandardScaler()
        X_t = scaler.fit(X).transform(X)
        np.testing.assert_allclose(X_t.mean(axis=0), 0, atol=1e-10)

    def test_constant_column_no_divide_by_zero(self):
        """Constant column must not cause NaN/Inf."""
        X = np.array([[5., 1.], [5., 2.], [5., 3.]])
        scaler = StreamingStandardScaler()
        X_t = scaler.fit(X).transform(X)
        assert np.all(np.isfinite(X_t)), "NaN/Inf in output for constant column"

    def test_nan_ignored_during_fit(self):
        """NaN values must be ignored when computing statistics."""
        X = np.array([[1., np.nan], [np.nan, 2.], [3., 4.]])
        scaler = StreamingStandardScaler()
        scaler.partial_fit(X)
        assert np.isfinite(scaler.mean_[0])
        assert np.isfinite(scaler.mean_[1])

    def test_nan_preserved_after_transform(self):
        """NaN values in X should be preserved in the transformed output."""
        X = np.array([[1., np.nan], [3., 4.]])
        scaler = StreamingStandardScaler()
        scaler.partial_fit(X)
        X_t = scaler.transform(X)
        assert np.isnan(X_t[0, 1])

    def test_transform_before_fit_raises(self):
        """transform() before partial_fit should raise ValueError."""
        scaler = StreamingStandardScaler()
        with pytest.raises(ValueError, match="partial_fit"):
            scaler.transform(np.array([[1., 2.]]))

    def test_feature_count_mismatch_raises(self):
        """Different number of features between fit and transform raises."""
        scaler = StreamingStandardScaler()
        scaler.partial_fit(np.array([[1., 2., 3.]]))
        with pytest.raises(ValueError):
            scaler.transform(np.array([[1., 2.]]))

    def test_fit_resets_state(self):
        """Calling fit() should reset previous partial_fit state."""
        X1 = np.array([[100., 200.]])
        X2 = np.array([[1., 2.], [3., 4.]])
        scaler = StreamingStandardScaler()
        scaler.partial_fit(X1)
        scaler.fit(X2)
        np.testing.assert_allclose(scaler.mean_, np.mean(X2, axis=0), rtol=1e-6)

    def test_empty_chunk_raises(self):
        """Zero-row input should raise ValueError."""
        scaler = StreamingStandardScaler()
        with pytest.raises(ValueError):
            scaler.partial_fit(np.empty((0, 3)))

    def test_single_sample_chunk(self):
        """Single-sample chunk should update without error."""
        scaler = StreamingStandardScaler()
        scaler.partial_fit(np.array([[5., 10.]]))
        assert scaler.n_samples_seen_[0] == 1


# ---------------------------------------------------------------------------
# StreamingMinMaxScaler
# ---------------------------------------------------------------------------

class TestStreamingMinMaxScaler:

    def test_transform_within_range(self):
        """All transformed values should lie within feature_range."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 3))
        scaler = StreamingMinMaxScaler(feature_range=(0, 1))
        X_t = scaler.fit(X).transform(X)
        assert X_t.min() >= -1e-9
        assert X_t.max() <= 1 + 1e-9

    def test_custom_range(self):
        """Custom feature_range should be respected."""
        X = np.array([[0., 10.], [5., 20.], [10., 30.]])
        scaler = StreamingMinMaxScaler(feature_range=(-1, 1))
        X_t = scaler.fit(X).transform(X)
        np.testing.assert_allclose(X_t[:, 0].min(), -1, atol=1e-9)
        np.testing.assert_allclose(X_t[:, 0].max(), 1, atol=1e-9)

    def test_incremental_range_expands(self):
        """Running min/max must expand monotonically as new chunks arrive."""
        X1 = np.array([[0., 5.], [3., 8.]])
        X2 = np.array([[-5., 15.]])
        scaler = StreamingMinMaxScaler()
        scaler.partial_fit(X1).partial_fit(X2)
        assert scaler.data_min_[0] <= -5
        assert scaler.data_max_[1] >= 15

    def test_constant_column(self):
        """Constant column should not produce NaN."""
        X = np.array([[3., 1.], [3., 2.], [3., 3.]])
        scaler = StreamingMinMaxScaler()
        X_t = scaler.fit(X).transform(X)
        assert np.all(np.isfinite(X_t))

    def test_invalid_feature_range_raises(self):
        """feature_range with low >= high should raise."""
        with pytest.raises(ValueError):
            StreamingMinMaxScaler(feature_range=(5, 5))

    def test_transform_before_fit_raises(self):
        scaler = StreamingMinMaxScaler()
        with pytest.raises(ValueError):
            scaler.transform(np.array([[1., 2.]]))

    def test_two_chunks_cover_full_range(self):
        """After two disjoint chunks, transform of combined data is in [0,1]."""
        X1 = np.array([[0.], [5.]])
        X2 = np.array([[10.], [20.]])
        scaler = StreamingMinMaxScaler()
        scaler.partial_fit(X1).partial_fit(X2)
        X_all = np.vstack([X1, X2])
        X_t = scaler.transform(X_all)
        assert X_t.min() >= -1e-9
        assert X_t.max() <= 1 + 1e-9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInternals:

    def test_gini_pure(self):
        """Pure class distribution should have Gini = 0."""
        assert _gini(np.array([10., 0.])) == pytest.approx(0.0)

    def test_gini_balanced(self):
        """50/50 split should have Gini = 0.5."""
        assert _gini(np.array([5., 5.])) == pytest.approx(0.5)

    def test_hoeffding_bound_decreases_with_n(self):
        """More samples means smaller Hoeffding bound."""
        eps10 = _hoeffding_bound(10, 1e-7)
        eps100 = _hoeffding_bound(100, 1e-7)
        assert eps10 > eps100

    def test_hoeffding_bound_zero_samples(self):
        """Zero samples should return range (worst case = 1.0)."""
        assert _hoeffding_bound(0, 1e-7) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# StreamingDecisionTree
# ---------------------------------------------------------------------------

class TestStreamingDecisionTree:

    def test_fit_predict_binary(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(min_samples_split=20, delta=1e-3)
        tree.fit(X, y)
        preds = tree.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.6, f"Expected accuracy > 0.6, got {acc:.3f}"

    def test_incremental_matches_batch(self, binary_dataset):
        """Accuracy after streaming same data in chunks ≥ single-chunk accuracy × 0.9."""
        X, y = binary_dataset
        tree_batch = StreamingDecisionTree(min_samples_split=20, delta=1e-3, random_state=0)
        tree_batch.fit(X, y)
        acc_batch = np.mean(tree_batch.predict(X) == y)

        tree_stream = StreamingDecisionTree(min_samples_split=20, delta=1e-3, random_state=0)
        chunk_size = 40
        for i in range(0, len(X), chunk_size):
            tree_stream.partial_fit(X[i:i+chunk_size], y[i:i+chunk_size],
                                    classes=np.array([0, 1]))
        acc_stream = np.mean(tree_stream.predict(X) == y)
        assert acc_stream >= acc_batch * 0.9, \
            f"Streaming acc {acc_stream:.3f} vs batch {acc_batch:.3f}"

    def test_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        tree = StreamingDecisionTree(min_samples_split=30, delta=1e-3)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_predict_proba_sums_to_one(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(min_samples_split=20)
        tree.fit(X, y)
        proba = tree.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_predict_before_fit_raises(self):
        tree = StreamingDecisionTree()
        with pytest.raises(ValueError):
            tree.predict(np.array([[1., 2.]]))

    def test_predict_wrong_features_raises(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(min_samples_split=20)
        tree.fit(X, y)
        with pytest.raises(ValueError):
            tree.predict(X[:, :2])

    def test_max_depth_respected(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(max_depth=2, min_samples_split=10, delta=1e-10)
        tree.fit(X, y)
        assert tree.depth_ <= 2

    def test_nan_in_features_handled(self):
        """Tree should train and predict without error when X contains NaN."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 3))
        y = (X[:, 0] > 0).astype(int)
        X[rng.random((100, 3)) < 0.1] = np.nan  # 10% missing
        tree = StreamingDecisionTree(min_samples_split=10)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert len(preds) == 100

    def test_all_same_class(self):
        """Single-class dataset should predict that class everywhere."""
        X = np.ones((50, 2))
        y = np.zeros(50, dtype=int)
        tree = StreamingDecisionTree(min_samples_split=5)
        tree.fit(X, y)
        assert np.all(tree.predict(X) == 0)

    def test_n_leaves_property(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(min_samples_split=20, delta=1e-3)
        tree.fit(X, y)
        assert tree.n_leaves_ >= 1

    def test_classes_from_argument(self):
        """Classes passed as argument should define the class set."""
        X = np.array([[0.], [1.], [2.]])
        y = np.array([0, 1, 0])
        tree = StreamingDecisionTree(min_samples_split=2)
        tree.partial_fit(X, y, classes=np.array([0, 1, 2]))
        assert 2 in tree.classes_

    def test_zero_variance_chunk_ok(self):
        """Zero-variance chunk (all identical features) should not raise."""
        X = np.ones((30, 2)) * 5.0
        y = np.array([0] * 15 + [1] * 15)
        tree = StreamingDecisionTree(min_samples_split=5)
        tree.partial_fit(X, y)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

class TestStreamingBaggingClassifier:

    def test_higher_accuracy_than_single_tree(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(min_samples_split=20, delta=1e-3, random_state=0)
        tree.fit(X, y)
        acc_tree = np.mean(tree.predict(X) == y)

        bag = StreamingBaggingClassifier(n_estimators=10, min_samples_split=20,
                                         delta=1e-3, random_state=0)
        bag.fit(X, y)
        acc_bag = np.mean(bag.predict(X) == y)
        # Ensemble should be at least as good as a single tree (on the training set)
        assert acc_bag >= acc_tree * 0.95

    def test_predict_proba_sums_to_one(self, binary_dataset):
        X, y = binary_dataset
        bag = StreamingBaggingClassifier(n_estimators=5, min_samples_split=20)
        bag.fit(X, y)
        proba = bag.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_streaming_partial_fit(self, binary_dataset):
        X, y = binary_dataset
        bag = StreamingBaggingClassifier(n_estimators=5, min_samples_split=20,
                                         random_state=1)
        for i in range(0, len(X), 50):
            bag.partial_fit(X[i:i+50], y[i:i+50], classes=np.array([0, 1]))
        preds = bag.predict(X)
        assert len(preds) == len(y)

    def test_n_estimators_invalid_raises(self):
        with pytest.raises(ValueError):
            StreamingBaggingClassifier(n_estimators=0)

    def test_predict_before_fit_raises(self):
        bag = StreamingBaggingClassifier(n_estimators=3)
        with pytest.raises(ValueError):
            bag.predict(np.array([[1., 2.]]))


class TestStreamingRandomForest:

    def test_accuracy_binary(self, binary_dataset):
        X, y = binary_dataset
        rf = StreamingRandomForest(n_estimators=10, max_features='sqrt',
                                   min_samples_split=20, delta=1e-3, random_state=42)
        rf.fit(X, y)
        acc = np.mean(rf.predict(X) == y)
        assert acc > 0.6

    def test_sqrt_features(self, binary_dataset):
        X, y = binary_dataset
        rf = StreamingRandomForest(n_estimators=5, max_features='sqrt', random_state=0)
        rf.fit(X, y)
        expected_k = max(1, int(np.sqrt(X.shape[1])))
        for subset in rf._feature_subsets:
            assert len(subset) == expected_k

    def test_log2_features(self, binary_dataset):
        X, y = binary_dataset
        rf = StreamingRandomForest(n_estimators=5, max_features='log2', random_state=0)
        rf.fit(X, y)
        expected_k = max(1, int(np.log2(X.shape[1])))
        for subset in rf._feature_subsets:
            assert len(subset) == expected_k

    def test_all_features_degenerates_to_bagging(self, binary_dataset):
        X, y = binary_dataset
        rf = StreamingRandomForest(n_estimators=5, max_features='all', random_state=0)
        rf.fit(X, y)
        for subset in rf._feature_subsets:
            assert len(subset) == X.shape[1]

    def test_invalid_max_features_raises(self, binary_dataset):
        X, y = binary_dataset
        rf = StreamingRandomForest(n_estimators=3, max_features='bad_value')
        with pytest.raises(ValueError):
            rf.fit(X, y)

    def test_predict_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        rf = StreamingRandomForest(n_estimators=5, min_samples_split=20)
        rf.fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(X), 2)


# ---------------------------------------------------------------------------
# Visualise module (smoke tests — check no exceptions raised)
# ---------------------------------------------------------------------------

class TestVisualise:

    def test_plot_metrics_runs(self):
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = visualise.plot_metrics({'accuracy': [0.5, 0.7, 0.8]})
        assert fig is not None

    def test_plot_metrics_empty_raises(self):
        with pytest.raises(ValueError):
            visualise.plot_metrics({})

    def test_plot_confusion_matrix_runs(self):
        import matplotlib
        matplotlib.use('Agg')
        cm = np.array([[10, 2], [3, 15]])
        fig, ax = visualise.plot_confusion_matrix(cm)
        assert fig is not None

    def test_plot_confusion_matrix_not_square_raises(self):
        with pytest.raises(ValueError):
            visualise.plot_confusion_matrix(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_plot_roc_curve_runs(self):
        import matplotlib
        matplotlib.use('Agg')
        fpr = np.array([0., 0.5, 1.])
        tpr = np.array([0., 0.8, 1.])
        fig, ax = visualise.plot_roc_curve(fpr, tpr, auc_score=0.9)
        assert fig is not None

    def test_plot_comparison_runs(self):
        import matplotlib
        matplotlib.use('Agg')
        data = {
            'Tree': {'accuracy': [0.6, 0.7, 0.8]},
            'Forest': {'accuracy': [0.65, 0.75, 0.85]},
        }
        fig, ax = visualise.plot_comparison(data)
        assert fig is not None

    def test_plot_comparison_missing_metric_raises(self):
        data = {
            'Tree': {'accuracy': [0.6, 0.7]},
        }
        with pytest.raises(ValueError):
            visualise.plot_comparison(data, metric_name='f1')

    def test_plot_tree_structure_empty(self):
        tree = StreamingDecisionTree()
        result = visualise.plot_tree_structure(tree)
        assert "empty" in result.lower()

    def test_plot_tree_structure_fitted(self, binary_dataset):
        X, y = binary_dataset
        tree = StreamingDecisionTree(min_samples_split=20, delta=1e-3)
        tree.fit(X, y)
        result = visualise.plot_tree_structure(tree)
        assert "LEAF" in result or "SPLIT" in result


# ---------------------------------------------------------------------------
# Integration: scaler + tree pipeline
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_scaler_then_tree_pipeline(self, binary_dataset):
        """Scale chunks then feed into tree — accuracy must exceed 60%."""
        X, y = binary_dataset
        scaler = StreamingStandardScaler()
        tree = StreamingDecisionTree(min_samples_split=20, delta=1e-3, random_state=0)
        classes = np.array([0, 1])

        chunk_size = 50
        for i in range(0, len(X), chunk_size):
            Xc = X[i:i+chunk_size]
            yc = y[i:i+chunk_size]
            scaler.partial_fit(Xc)
            Xc_scaled = scaler.transform(Xc)
            tree.partial_fit(Xc_scaled, yc, classes=classes)

        X_scaled_full = scaler.transform(X)
        acc = np.mean(tree.predict(X_scaled_full) == y)
        assert acc > 0.6, f"Pipeline accuracy = {acc:.3f}"

    def test_forest_streaming_pipeline(self, binary_dataset):
        """Full streaming pipeline with RandomForest must train without errors."""
        X, y = binary_dataset
        scaler = StreamingMinMaxScaler()
        rf = StreamingRandomForest(n_estimators=5, min_samples_split=20,
                                   max_features='sqrt', random_state=99)
        classes = np.array([0, 1])

        for i in range(0, len(X), 50):
            Xc = X[i:i+50]
            yc = y[i:i+50]
            scaler.partial_fit(Xc)
            rf.partial_fit(scaler.transform(Xc), yc, classes=classes)

        preds = rf.predict(scaler.transform(X))
        assert len(preds) == len(y)
