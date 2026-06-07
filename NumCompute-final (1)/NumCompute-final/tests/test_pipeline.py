import numpy as np
import pytest

from numcompute.pipeline import FeatureUnion, Pipeline
from numcompute.preprocessing import MinMaxScaler, StandardScaler


class DummyModel:
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        return X[:, 0]


class ColumnSlice:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.cols]


def test_pipeline_fit_predict():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("model", DummyModel())])
    pipe.fit(X)
    preds = pipe.predict(X)
    assert preds.shape == (4,)
    assert np.allclose(preds, StandardScaler().fit_transform(X)[:, 0])


def test_pipeline_transform_only():
    X = np.array([[1.0], [2.0], [3.0]])
    pipe = Pipeline([("scale", MinMaxScaler())])
    out = pipe.fit_transform(X)
    assert np.min(out) == 0.0
    assert np.max(out) == 1.0


def test_feature_union_concatenates():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    union = FeatureUnion(
        [
            ("col0", ColumnSlice([0])),
            ("col1", ColumnSlice([1])),
        ]
    )
    out = union.fit_transform(X)
    assert out.shape == (3, 2)
    assert np.array_equal(out, X)


def test_pipeline_predict_without_model_raises():
    X = np.array([[1.0], [2.0]])
    pipe = Pipeline([("scale", StandardScaler())]).fit(X)
    with pytest.raises(ValueError):
        pipe.predict(X)

def test_invalid_step_format():
    import pytest
    with pytest.raises(ValueError):
        Pipeline([("only_name",)])        

class BadStep:
    pass

def test_missing_fit_raises():
    import pytest
    with pytest.raises(ValueError):
        Pipeline([("bad", BadStep())])

class NoTransform:
    def fit(self, X, y=None):
        return self

def test_missing_transform_in_middle():
    import pytest
    pipe = Pipeline([("bad", NoTransform()), ("model", DummyModel())])
    with pytest.raises(ValueError):
        pipe.fit(np.array([[1.0]]))

class BadTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([1, 2, 3])  # 1D

def test_feature_union_invalid_shape():
    import pytest
    union = FeatureUnion([("bad", BadTransformer())])
    with pytest.raises(ValueError):
        union.fit_transform(np.array([[1], [2], [3]]))        
