import numpy as np
import pytest

from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def test_standard_scaler_mean_zero_std_one():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0.0)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0)


def test_standard_scaler_constant_column_becomes_zero():
    X = np.array([[1, 2], [1, 4], [1, 6]], dtype=float)
    scaler = StandardScaler().fit(X)
    out = scaler.transform(X)

    assert np.allclose(out[:, 0], 0.0)


def test_standard_scaler_preserves_nan():
    X = np.array([[1, np.nan], [3, 4], [5, 6]], dtype=float)
    out = StandardScaler().fit_transform(X)

    assert np.isnan(out[0, 1])


def test_standard_scaler_transform_before_fit_raises():
    with pytest.raises(ValueError):
        StandardScaler().transform(np.array([[1, 2]], dtype=float))


def test_standard_scaler_feature_mismatch_raises():
    scaler = StandardScaler().fit(np.array([[1, 2], [3, 4]], dtype=float))

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1, 2, 3]], dtype=float))


def test_minmax_scaler_default_range():
    X = np.array([[1], [2], [3]], dtype=float)
    X_scaled = MinMaxScaler().fit_transform(X)

    assert np.min(X_scaled) == 0.0
    assert np.max(X_scaled) == 1.0


def test_minmax_scaler_custom_range():
    X = np.array([[1], [2], [3]], dtype=float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    assert np.min(X_scaled) == -1.0
    assert np.max(X_scaled) == 1.0


def test_minmax_scaler_constant_column_maps_to_lower_bound():
    X = np.array([[5], [5], [5]], dtype=float)
    out = MinMaxScaler(feature_range=(2, 4)).fit_transform(X)

    assert np.allclose(out, np.full((3, 1), 2.0))


def test_minmax_invalid_feature_range_raises():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 1))


def test_simple_imputer_mean_removes_nan():
    X = np.array([[1, np.nan], [3, 4]], dtype=float)
    imputer = SimpleImputer()
    X_filled = imputer.fit_transform(X)

    assert not np.isnan(X_filled).any()
    assert X_filled[0, 1] == 4.0


def test_simple_imputer_median():
    X = np.array([[1, np.nan], [100, 4], [3, 6]], dtype=float)
    out = SimpleImputer(strategy="median").fit_transform(X)

    assert out[0, 1] == 5.0


def test_simple_imputer_constant():
    X = np.array([[1, np.nan], [3, 4]], dtype=float)
    imputer = SimpleImputer(strategy="constant", fill_value=-5)
    X_filled = imputer.fit_transform(X)

    assert X_filled[0, 1] == -5


def test_simple_imputer_all_nan_column_uses_fill_value():
    X = np.array([[np.nan, 1], [np.nan, 2]], dtype=float)
    out = SimpleImputer(strategy="mean", fill_value=0).fit_transform(X)

    assert np.allclose(out[:, 0], np.array([0.0, 0.0]))


def test_simple_imputer_invalid_strategy_raises():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="bad")


def test_one_hot_encoder_single_column_shape_and_values():
    X = np.array([["A"], ["B"], ["A"]], dtype=object)
    encoder = OneHotEncoder()
    X_enc = encoder.fit_transform(X)

    assert X_enc.shape == (3, 2)
    assert np.array_equal(X_enc, np.array([[1, 0], [0, 1], [1, 0]]))


def test_one_hot_encoder_multiple_columns():
    X = np.array([["A", "X"], ["B", "X"], ["A", "Y"]], dtype=object)
    encoder = OneHotEncoder()
    X_enc = encoder.fit_transform(X)

    assert X_enc.shape == (3, 4)
    assert np.all(X_enc.sum(axis=1) == 2)


def test_one_hot_encoder_unknown_category_raises():
    X = np.array([["A"], ["B"]], dtype=object)
    encoder = OneHotEncoder().fit(X)

    with pytest.raises(ValueError):
        encoder.transform(np.array([["C"]], dtype=object))


def test_one_hot_encoder_unknown_category_ignore():
    X = np.array([["A"], ["B"]], dtype=object)
    encoder = OneHotEncoder(handle_unknown="ignore").fit(X)
    out = encoder.transform(np.array([["C"]], dtype=object))

    assert np.array_equal(out, np.array([[0, 0]]))


def test_one_hot_encoder_transform_before_fit_raises():
    with pytest.raises(ValueError):
        OneHotEncoder().transform(np.array([["A"]], dtype=object))


def test_preprocessors_require_2d_input():
    with pytest.raises(ValueError):
        StandardScaler().fit(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        MinMaxScaler().fit(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        SimpleImputer().fit(np.array([1, 2, np.nan]))

    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.array(["A", "B"]))
