"""Pipeline abstractions for chaining transformers and estimators."""

from __future__ import annotations

import numpy as np


class Pipeline:
    """
    Sequentially apply a list of transforms and a final estimator.

    Each intermediate step must implement fit and transform.
    The final step may implement fit/transform or fit/predict.
    """

    def __init__(self, steps):
        if not isinstance(steps, (list, tuple)) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list of (name, step) pairs.")
        self.steps = list(steps)
        self._validate_steps()

    def _validate_steps(self):
        for item in self.steps:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each step must be a (name, transformer) pair.")
            name, step = item
            if not isinstance(name, str) or not name:
                raise ValueError("Each step name must be a non-empty string.")
            if not hasattr(step, "fit"):
                raise ValueError(f"Step '{name}' must implement fit().")

    @property
    def named_steps(self):
        return dict(self.steps)

    @staticmethod
    def _fit_step(step, X, y=None):
        try:
            return step.fit(X, y)
        except TypeError:
            return step.fit(X)

    @staticmethod
    def _fit_transform_step(step, X, y=None):
        if hasattr(step, "fit_transform"):
            try:
                return step.fit_transform(X, y)
            except TypeError:
                return step.fit_transform(X)
        fitted = Pipeline._fit_step(step, X, y)
        return fitted.transform(X)

    def fit(self, X, y=None):
        Xt = X
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Intermediate step '{name}' must implement transform().")
            Xt = self._fit_transform_step(step, Xt, y)
        _, final_step = self.steps[-1]
        self._fit_step(final_step, Xt, y)
        return self

    def transform(self, X):
        Xt = X
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' does not implement transform().")
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        Xt = X
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Intermediate step '{name}' must implement transform().")
            Xt = self._fit_transform_step(step, Xt, y)
        last_name, last_step = self.steps[-1]
        if hasattr(last_step, "transform"):
            return self._fit_transform_step(last_step, Xt, y)
        raise ValueError(f"Final step '{last_name}' does not implement transform().")

    def predict(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Intermediate step '{name}' must implement transform().")
            Xt = step.transform(Xt)
        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "predict"):
            raise ValueError(f"Final step '{final_name}' must implement predict().")
        return final_step.predict(Xt)


class FeatureUnion:
    """
    Concatenate outputs of multiple transformers column-wise.
    """

    def __init__(self, transformer_list):
        if not isinstance(transformer_list, (list, tuple)) or len(transformer_list) == 0:
            raise ValueError(
                "transformer_list must be a non-empty list of (name, transformer) pairs."
            )
        self.transformer_list = list(transformer_list)
        for name, transformer in self.transformer_list:
            if not isinstance(name, str) or not name:
                raise ValueError("Each transformer name must be a non-empty string.")
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise ValueError(f"Transformer '{name}' must implement fit() and transform().")

    def fit(self, X, y=None):
        for _, transformer in self.transformer_list:
            Pipeline._fit_step(transformer, X, y)
        return self

    def transform(self, X):
        transformed = [np.asarray(transformer.transform(X)) for _, transformer in self.transformer_list]
        if any(arr.ndim != 2 for arr in transformed):
            raise ValueError("All transformed outputs must be 2D arrays.")
        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
