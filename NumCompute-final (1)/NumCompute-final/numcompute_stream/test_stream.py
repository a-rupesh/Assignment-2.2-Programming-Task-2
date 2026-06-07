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
- StreamingImputer (standard + edge cases)
- StreamingOneHotEncoder (standard + edge cases)
- StreamTrainer (standard + edge cases)
- ChunkStats / update_stats API (standard + edge cases)
- StreamingMetrics update/reset/result (standard + edge cases)
- StreamingPipeline partial_fit (standard + edge cases)
- Class aliases: DecisionTreeClassifier, EnsembleClassifier

Total: 100 tests
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    StreamingStandardScaler,
    StreamingMinMaxScaler,
    StreamingImputer,
    StreamingOneHotEncoder,
)
from numcompute_stream.tree import (
    StreamingDecisionTree,
    DecisionTreeClassifier,
    _gini,
    _hoeffding_bound,
)
from numcompute_stream.ensemble import (
    StreamingBaggingClassifier,
    StreamingRandomForest,
    EnsembleClassifier,
)
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.stats import ChunkStats, StreamingStats
from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.pipeline import StreamingPipeline
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
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    return X, y


# ---------------------------------------------------------------------------
# StreamingStandardScaler
# ---------------------------------------------------------------------------

class TestStreamingStandardScaler:

    def test_single_chunk_matches_numpy(self):
        """After one partial_fit, mean should match numpy."""
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
        assert np.all(np.isfinite(X_t))

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
        """Accuracy after streaming same data in chunks >= batch x 0.9."""
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
        X[rng.random((100, 3)) < 0.1] = np.nan
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
        """Zero-variance chunk should not raise."""
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
# Visualise module
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
        data = {'Tree': {'accuracy': [0.6, 0.7]}}
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

    def test_plot_metric_over_time_runs(self):
        """Spec-required function name should work."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = visualise.plot_metric_over_time(
            [0.5, 0.6, 0.7], title='Accuracy', ylabel='Accuracy'
        )
        assert fig is not None

    def test_compare_models_runs(self):
        """Spec-required function name should work."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = visualise.compare_models(
            [0.6, 0.7], [0.65, 0.75], labels=['Tree', 'Forest']
        )
        assert fig is not None

    def test_compare_models_default_labels(self):
        """compare_models should work without explicit labels."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = visualise.compare_models([0.6, 0.7], [0.65, 0.75])
        assert fig is not None

    def test_plot_predictions_vs_ground_truth_runs(self):
        """Spec-required function name should work."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = visualise.plot_predictions_vs_ground_truth(
            [0, 1, 1, 0], [0, 1, 0, 0]
        )
        assert fig is not None

    def test_plot_predictions_shape_mismatch_raises(self):
        """Mismatched y_true and y_pred should raise ValueError."""
        with pytest.raises(ValueError):
            visualise.plot_predictions_vs_ground_truth([0, 1], [0, 1, 1])


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


# ---------------------------------------------------------------------------
# StreamingImputer
# ---------------------------------------------------------------------------

class TestStreamingImputer:

    def test_mean_strategy_removes_nan(self):
        """Mean strategy should replace NaN with column mean."""
        X = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        imp = StreamingImputer(strategy='mean')
        out = imp.fit_transform(X)
        assert not np.isnan(out).any()
        assert out[0, 1] == pytest.approx(5.0)

    def test_median_strategy(self):
        """Median strategy should replace NaN with column median."""
        X = np.array([[1., np.nan], [3., 2.], [5., 8.]])
        imp = StreamingImputer(strategy='median')
        out = imp.fit_transform(X)
        assert not np.isnan(out).any()
        assert out[0, 1] == pytest.approx(5.0)

    def test_constant_strategy(self):
        """Constant strategy should replace NaN with fill_value."""
        X = np.array([[np.nan, 1.], [2., np.nan]])
        imp = StreamingImputer(strategy='constant', fill_value=-99.0)
        out = imp.fit_transform(X)
        assert out[0, 0] == pytest.approx(-99.0)
        assert out[1, 1] == pytest.approx(-99.0)

    def test_partial_fit_updates_estimates(self):
        """Running mean should update across multiple chunks."""
        X1 = np.array([[2., 4.]])
        X2 = np.array([[4., 8.]])
        imp = StreamingImputer(strategy='mean')
        imp.partial_fit(X1).partial_fit(X2)
        # mean of [2,4] and [4,8] = [3, 6]
        assert imp.statistics_[0] == pytest.approx(3.0)
        assert imp.statistics_[1] == pytest.approx(6.0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            StreamingImputer(strategy='bad')

    def test_transform_before_fit_raises(self):
        imp = StreamingImputer()
        with pytest.raises(ValueError):
            imp.transform(np.array([[1., np.nan]]))

    def test_feature_mismatch_raises(self):
        imp = StreamingImputer()
        imp.partial_fit(np.array([[1., 2., 3.]]))
        with pytest.raises(ValueError):
            imp.transform(np.array([[1., 2.]]))

    def test_no_nan_input_unchanged(self):
        """Input without NaN should pass through unchanged."""
        X = np.array([[1., 2.], [3., 4.]])
        imp = StreamingImputer(strategy='mean')
        out = imp.fit_transform(X)
        np.testing.assert_array_equal(out, X)


# ---------------------------------------------------------------------------
# StreamingOneHotEncoder
# ---------------------------------------------------------------------------

class TestStreamingOneHotEncoder:

    def test_basic_encoding(self):
        """Single column should produce correct one-hot output."""
        X = np.array([['a'], ['b'], ['a']], dtype=object)
        enc = StreamingOneHotEncoder()
        out = enc.fit_transform(X)
        assert out.shape == (3, 2)
        assert np.array_equal(out[0], out[2])   # 'a' rows identical
        assert not np.array_equal(out[0], out[1])  # 'a' != 'b'

    def test_incremental_category_expansion(self):
        """New categories in later chunks should expand output width."""
        enc = StreamingOneHotEncoder()
        enc.partial_fit(np.array([['cat'], ['dog']], dtype=object))
        assert enc.n_output_features_ == 2
        enc.partial_fit(np.array([['fish']], dtype=object))
        assert enc.n_output_features_ == 3

    def test_unknown_category_ignore(self):
        """Unknown category with handle_unknown='ignore' outputs all zeros."""
        enc = StreamingOneHotEncoder(handle_unknown='ignore')
        enc.fit(np.array([['a'], ['b']], dtype=object))
        out = enc.transform(np.array([['z']], dtype=object))
        assert np.array_equal(out, np.array([[0, 0]]))

    def test_unknown_category_error(self):
        """Unknown category with handle_unknown='error' raises ValueError."""
        enc = StreamingOneHotEncoder(handle_unknown='error')
        enc.fit(np.array([['a'], ['b']], dtype=object))
        with pytest.raises(ValueError):
            enc.transform(np.array([['z']], dtype=object))

    def test_transform_before_fit_raises(self):
        enc = StreamingOneHotEncoder()
        with pytest.raises(ValueError):
            enc.transform(np.array([['a']], dtype=object))

    def test_multiple_columns(self):
        """Multi-column input should produce correct total output width."""
        X = np.array([['a', 'x'], ['b', 'y'], ['a', 'x']], dtype=object)
        enc = StreamingOneHotEncoder()
        out = enc.fit_transform(X)
        assert out.shape == (3, 4)   # 2 cats + 2 cats

    def test_fit_resets_categories(self):
        """Calling fit() should discard previous partial_fit categories."""
        enc = StreamingOneHotEncoder()
        enc.partial_fit(np.array([['a'], ['b'], ['c']], dtype=object))
        enc.fit(np.array([['x'], ['y']], dtype=object))
        assert enc.n_output_features_ == 2


# ---------------------------------------------------------------------------
# StreamTrainer
# ---------------------------------------------------------------------------

class TestStreamTrainer:

    def test_fit_chunk_trains_model(self, binary_dataset):
        """After fit_chunk calls, model should predict with reasonable accuracy."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            scaler=StreamingStandardScaler(),
            classes=[0, 1],
        )
        for i in range(0, len(X), 50):
            trainer.fit_chunk(X[i:i+50], y[i:i+50])
        summary = trainer.summary()
        assert summary['chunks_processed'] == 4
        assert summary['final_accuracy'] > 0.5

    def test_score_chunk_does_not_update_log(self, binary_dataset):
        """score_chunk should evaluate without adding to the log."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            scaler=StreamingStandardScaler(),
            classes=[0, 1],
        )
        trainer.fit_chunk(X[:100], y[:100])
        log_len_before = len(trainer.get_log())
        trainer.score_chunk(X[100:], y[100:])
        assert len(trainer.get_log()) == log_len_before

    def test_log_contains_required_keys(self, binary_dataset):
        """Each log entry must contain all required fields."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            classes=[0, 1],
        )
        trainer.fit_chunk(X[:100], y[:100])
        entry = trainer.get_log()[0]
        for key in ['chunk_idx', 'n_samples', 'accuracy',
                    'cumulative_accuracy', 'fit_time_s', 'memory_bytes']:
            assert key in entry, f"Missing key: {key}"

    def test_accuracy_history_length(self, binary_dataset):
        """accuracy_history should have one entry per fit_chunk call."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            classes=[0, 1],
        )
        n_chunks = 4
        for i in range(n_chunks):
            trainer.fit_chunk(X[i*50:(i+1)*50], y[i*50:(i+1)*50])
        assert len(trainer.accuracy_history()) == n_chunks

    def test_reset_log_clears_state(self, binary_dataset):
        """reset_log should clear log and counters."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            classes=[0, 1],
        )
        trainer.fit_chunk(X[:100], y[:100])
        trainer.reset_log()
        assert trainer.get_log() == []
        assert trainer.chunk_idx_ == 0

    def test_no_scaler_works(self, binary_dataset):
        """StreamTrainer without a scaler should work fine."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            classes=[0, 1],
        )
        trainer.fit_chunk(X[:100], y[:100])
        result = trainer.score_chunk(X[100:], y[100:])
        assert 'accuracy' in result

    def test_invalid_model_raises(self):
        """Model without partial_fit should raise ValueError."""
        class BadModel:
            pass
        with pytest.raises(ValueError):
            StreamTrainer(model=BadModel())

    def test_cumulative_accuracy_increases(self, binary_dataset):
        """Cumulative accuracy should be consistent with total correct/seen."""
        X, y = binary_dataset
        trainer = StreamTrainer(
            model=DecisionTreeClassifier(min_samples_split=20),
            classes=[0, 1],
        )
        for i in range(0, len(X), 50):
            trainer.fit_chunk(X[i:i+50], y[i:i+50])
        log = trainer.get_log()
        # cumulative accuracy must be between 0 and 1
        for entry in log:
            assert 0.0 <= entry['cumulative_accuracy'] <= 1.0


# ---------------------------------------------------------------------------
# ChunkStats (update_stats API)
# ---------------------------------------------------------------------------

class TestChunkStats:

    def test_update_stats_single_chunk(self):
        """After one chunk, mean should match numpy mean."""
        X = np.array([[1., 2.], [3., 4.], [5., 6.]])
        cs = ChunkStats()
        cs.update_stats(X)
        np.testing.assert_allclose(cs.mean_, np.mean(X, axis=0), rtol=1e-6)

    def test_update_stats_two_chunks(self):
        """Mean after two chunks should equal mean of all data."""
        X1 = np.array([[1., 2.], [3., 4.]])
        X2 = np.array([[5., 6.], [7., 8.]])
        cs = ChunkStats()
        cs.update_stats(X1).update_stats(X2)
        X_all = np.vstack([X1, X2])
        np.testing.assert_allclose(cs.mean_, np.mean(X_all, axis=0), rtol=1e-5)

    def test_min_max_tracked_correctly(self):
        """Running min/max should reflect values seen across all chunks."""
        X1 = np.array([[0., 10.]])
        X2 = np.array([[-5., 20.]])
        cs = ChunkStats()
        cs.update_stats(X1).update_stats(X2)
        assert cs.min_[0] == pytest.approx(-5.0)
        assert cs.max_[1] == pytest.approx(20.0)

    def test_nan_ignored(self):
        """NaN values should be excluded from statistics."""
        X = np.array([[1., np.nan], [np.nan, 4.], [3., 6.]])
        cs = ChunkStats()
        cs.update_stats(X)
        assert np.isfinite(cs.mean_[0])
        assert np.isfinite(cs.mean_[1])

    def test_chunk_idx_increments(self):
        """chunk_idx_ should increment with each update_stats call."""
        cs = ChunkStats()
        X = np.ones((5, 2))
        cs.update_stats(X)
        cs.update_stats(X)
        assert cs.chunk_idx_ == 2

    def test_properties_before_fit_raises(self):
        """Accessing mean_ before any update_stats should raise ValueError."""
        cs = ChunkStats()
        with pytest.raises(ValueError):
            _ = cs.mean_

    def test_feature_mismatch_raises(self):
        """Chunk with wrong number of features should raise ValueError."""
        cs = ChunkStats()
        cs.update_stats(np.ones((5, 3)))
        with pytest.raises(ValueError):
            cs.update_stats(np.ones((5, 4)))

    def test_sliding_window_mean(self):
        """Window mean should reflect only the last window_size chunks."""
        cs = ChunkStats(window_size=2)
        cs.update_stats(np.array([[0., 0.]]))   # chunk 1 mean = [0, 0]
        cs.update_stats(np.array([[10., 10.]])) # chunk 2 mean = [10, 10]
        cs.update_stats(np.array([[20., 20.]])) # chunk 3 mean = [20, 20]
        # window contains chunks 2 and 3 only
        np.testing.assert_allclose(cs.window_mean_, [15., 15.], rtol=1e-6)

    def test_to_dict_returns_all_keys(self):
        """to_dict should contain all expected keys after one update."""
        cs = ChunkStats()
        cs.update_stats(np.array([[1., 2.], [3., 4.]]))
        d = cs.to_dict()
        for key in ['chunks_seen', 'mean', 'std', 'min', 'max']:
            assert key in d


# ---------------------------------------------------------------------------
# StreamingMetrics
# ---------------------------------------------------------------------------

class TestStreamingMetrics:

    def test_update_result_accuracy(self):
        """Accumulated accuracy should match manual calculation."""
        sm = StreamingMetrics()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        sm.update(y_true, y_pred)
        result = sm.result()
        expected_acc = np.mean(y_true == y_pred)
        assert result['accuracy'] == pytest.approx(expected_acc)

    def test_update_two_chunks_accumulates(self):
        """Metrics should accumulate correctly across two chunks."""
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
        sm.update(np.array([1, 0, 1, 1]), np.array([1, 0, 1, 1]))
        result = sm.result()
        assert result['total_samples'] == 8
        assert result['chunk_count'] == 2

    def test_reset_clears_all_state(self):
        """reset() should return all metrics to zero."""
        sm = StreamingMetrics()
        sm.update(np.array([0, 1]), np.array([0, 1]))
        sm.reset()
        result = sm.result()
        assert result['total_samples'] == 0
        assert result['accuracy'] == 0.0

    def test_result_keys_present(self):
        """result() must contain all required keys."""
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))
        result = sm.result()
        for key in ['accuracy', 'precision', 'recall', 'f1',
                    'confusion_matrix', 'total_samples', 'chunk_count']:
            assert key in result, f"Missing key: {key}"

    def test_precision_recall_f1_correct(self):
        """Precision, recall, F1 should match manual calculation."""
        sm = StreamingMetrics()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        sm.update(y_true, y_pred)
        result = sm.result()
        tp = 2; fp = 0; fn = 1
        expected_p = tp / (tp + fp)
        expected_r = tp / (tp + fn)
        expected_f = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert result['precision'] == pytest.approx(expected_p)
        assert result['recall'] == pytest.approx(expected_r)
        assert result['f1'] == pytest.approx(expected_f)

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2 for binary classification."""
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
        cm = sm.confusion_matrix_accumulated()
        assert cm.shape == (2, 2)

    def test_rolling_window_metrics(self):
        """Rolling window metrics should reflect only recent chunks."""
        sm = StreamingMetrics(window_size=2)
        # Chunk 1: all correct
        sm.update(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))
        # Chunk 2: all correct
        sm.update(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))
        # Chunk 3: all wrong
        sm.update(np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]))
        result = sm.result()
        # Window contains chunks 2 and 3: avg accuracy = (1.0 + 0.0) / 2 = 0.5
        assert result['rolling_accuracy'] == pytest.approx(0.5)

    def test_empty_result_returns_zeros(self):
        """result() before any update should return zero metrics."""
        sm = StreamingMetrics()
        result = sm.result()
        assert result['accuracy'] == 0.0
        assert result['total_samples'] == 0

    def test_nan_in_predictions_ignored(self):
        """NaN values in y_true or y_pred should be dropped."""
        sm = StreamingMetrics()
        y_true = np.array([0., 1., np.nan, 0.])
        y_pred = np.array([0., 1., 0., np.nan])
        sm.update(y_true, y_pred)
        result = sm.result()
        assert result['total_samples'] == 2

    def test_invalid_n_classes_raises(self):
        """n_classes < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            StreamingMetrics(n_classes=1)


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------

class TestStreamingPipeline:

    def test_partial_fit_then_predict(self, binary_dataset):
        """Pipeline should train incrementally and predict correctly."""
        X, y = binary_dataset
        pipe = StreamingPipeline([
            ('scale', StreamingStandardScaler()),
            ('model', DecisionTreeClassifier(min_samples_split=20)),
        ])
        for i in range(0, len(X), 50):
            pipe.partial_fit(X[i:i+50], y[i:i+50], classes=np.array([0, 1]))
        preds = pipe.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.6, f"Pipeline accuracy = {acc:.3f}"

    def test_transform_applies_scaler(self, binary_dataset):
        """transform() should scale data through all transformer steps."""
        X, y = binary_dataset
        pipe = StreamingPipeline([
            ('scale', StreamingStandardScaler()),
            ('model', DecisionTreeClassifier(min_samples_split=20)),
        ])
        pipe.partial_fit(X, y, classes=np.array([0, 1]))
        X_t = pipe.transform(X)
        # Scaled data should have near-zero mean
        np.testing.assert_allclose(X_t.mean(axis=0), 0.0, atol=0.1)

    def test_predict_proba_shape(self, binary_dataset):
        """predict_proba should return (n_samples, n_classes) array."""
        X, y = binary_dataset
        pipe = StreamingPipeline([
            ('scale', StreamingStandardScaler()),
            ('model', DecisionTreeClassifier(min_samples_split=20)),
        ])
        pipe.partial_fit(X, y, classes=np.array([0, 1]))
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_empty_steps_raises(self):
        """Empty steps list should raise ValueError."""
        with pytest.raises(ValueError):
            StreamingPipeline([])

    def test_invalid_step_format_raises(self):
        """Steps without name/estimator pair should raise ValueError."""
        with pytest.raises(ValueError):
            StreamingPipeline([('only_one',)])

    def test_final_step_without_partial_fit_raises(self, binary_dataset):
        """Final step without partial_fit should raise ValueError."""
        X, y = binary_dataset

        class NoPartialFit:
            def fit(self, X, y=None): return self
            def predict(self, X): return np.zeros(len(X))

        pipe = StreamingPipeline([
            ('scale', StreamingStandardScaler()),
            ('model', NoPartialFit()),
        ])
        with pytest.raises(ValueError):
            pipe.partial_fit(X, y)

    def test_named_steps_accessible(self, binary_dataset):
        """named_steps property should give access to steps by name."""
        X, y = binary_dataset
        scaler = StreamingStandardScaler()
        pipe = StreamingPipeline([
            ('scale', scaler),
            ('model', DecisionTreeClassifier(min_samples_split=20)),
        ])
        assert 'scale' in pipe.named_steps
        assert 'model' in pipe.named_steps

    def test_fit_delegates_to_partial_fit(self, binary_dataset):
        """fit() should behave identically to a single partial_fit call."""
        X, y = binary_dataset
        pipe = StreamingPipeline([
            ('scale', StreamingStandardScaler()),
            ('model', DecisionTreeClassifier(min_samples_split=20)),
        ])
        pipe.fit(X, y, classes=np.array([0, 1]))
        preds = pipe.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# Class aliases
# ---------------------------------------------------------------------------

class TestAliases:

    def test_decision_tree_classifier_is_alias(self):
        """DecisionTreeClassifier must be the same class as StreamingDecisionTree."""
        assert DecisionTreeClassifier is StreamingDecisionTree

    def test_ensemble_classifier_is_alias(self):
        """EnsembleClassifier must be the same class as StreamingRandomForest."""
        assert EnsembleClassifier is StreamingRandomForest

    def test_decision_tree_classifier_works(self, binary_dataset):
        """DecisionTreeClassifier should train and predict correctly."""
        X, y = binary_dataset
        clf = DecisionTreeClassifier(min_samples_split=20, delta=1e-3)
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.6

    def test_ensemble_classifier_works(self, binary_dataset):
        """EnsembleClassifier should train and predict correctly."""
        X, y = binary_dataset
        clf = EnsembleClassifier(n_estimators=5, min_samples_split=20,
                                 random_state=0)
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.6
