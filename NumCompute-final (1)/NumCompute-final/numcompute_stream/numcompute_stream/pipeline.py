"""Streaming-compatible Pipeline for NumCompute Stream.

Extends the base Pipeline concept with ``partial_fit()`` so the entire
chain — scalers and model — can be updated incrementally chunk by chunk.

Classes
-------
StreamingPipeline
    Chains transformers and a final estimator; all steps support
    ``partial_fit()``.

Example
-------
    from numcompute_stream.pipeline import StreamingPipeline
    from numcompute_stream.preprocessing import StreamingStandardScaler
    from numcompute_stream.tree import DecisionTreeClassifier

    pipe = StreamingPipeline([
        ('scale', StreamingStandardScaler()),
        ('model', DecisionTreeClassifier(min_samples_split=20)),
    ])

    for X_chunk, y_chunk in stream:
        pipe.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))

    preds = pipe.predict(X_test)
"""

from __future__ import annotations

import numpy as np


class StreamingPipeline:
    """Chain of streaming transformers and a final streaming estimator.

    Each step except the last must implement ``partial_fit(X)`` and
    ``transform(X)``. The final step must implement
    ``partial_fit(X, y)`` and ``predict(X)``.

    Parameters
    ----------
    steps : list of (str, estimator) tuples
        Named pipeline steps. The last step is the model; all others
        are transformers.

    Examples
    --------
    >>> pipe = StreamingPipeline([
    ...     ('scaler', StreamingStandardScaler()),
    ...     ('model',  DecisionTreeClassifier()),
    ... ])
    >>> pipe.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
    >>> pipe.predict(X_test)
    """

    def __init__(self, steps: list):
        if not isinstance(steps, (list, tuple)) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list of (name, step) pairs.")
        self.steps = list(steps)
        self._validate()

    def _validate(self) -> None:
        for item in self.steps:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each step must be a (name, estimator) pair.")
            name, step = item
            if not isinstance(name, str) or not name:
                raise ValueError("Step names must be non-empty strings.")
            if not hasattr(step, 'fit') and not hasattr(step, 'partial_fit'):
                raise ValueError(f"Step '{name}' must implement fit() or partial_fit().")

    @property
    def named_steps(self) -> dict:
        return dict(self.steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform_steps(self):
        """Return all steps except the last."""
        return self.steps[:-1]

    def _final_step(self):
        return self.steps[-1]

    def _partial_fit_transformer(self, step, X, y=None):
        if hasattr(step, 'partial_fit'):
            try:
                step.partial_fit(X, y)
            except TypeError:
                step.partial_fit(X)
        elif hasattr(step, 'fit'):
            try:
                step.fit(X, y)
            except TypeError:
                step.fit(X)
        return step

    def _transform(self, step, X):
        if not hasattr(step, 'transform'):
            raise ValueError(f"Step does not implement transform().")
        return step.transform(X)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partial_fit(self, X, y=None, **kwargs) -> "StreamingPipeline":
        """Incrementally fit all steps on a new chunk.

        Transformers are updated with ``partial_fit(X)`` then the
        transformed data is passed to the next step. The final estimator
        is updated with ``partial_fit(X_transformed, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None
        **kwargs
            Passed to the final estimator's ``partial_fit`` (e.g. ``classes``).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        Xt = X

        # Update and transform through all intermediate steps
        for name, step in self._transform_steps():
            self._partial_fit_transformer(step, Xt)
            Xt = self._transform(step, Xt)

        # Update final estimator
        final_name, final_step = self._final_step()
        if not hasattr(final_step, 'partial_fit'):
            raise ValueError(
                f"Final step '{final_name}' must implement partial_fit()."
            )
        if y is not None:
            final_step.partial_fit(Xt, y, **kwargs)
        else:
            final_step.partial_fit(Xt, **kwargs)

        return self

    def fit(self, X, y=None, **kwargs) -> "StreamingPipeline":
        """Fit all steps on a single batch (delegates to partial_fit)."""
        return self.partial_fit(X, y, **kwargs)

    def transform(self, X) -> np.ndarray:
        """Apply all transformer steps (not the final estimator).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """
        Xt = np.asarray(X, dtype=float)
        for name, step in self._transform_steps():
            Xt = self._transform(step, Xt)
        return Xt

    def predict(self, X) -> np.ndarray:
        """Transform X through all steps then predict with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        Xt = self.transform(X)
        final_name, final_step = self._final_step()
        if not hasattr(final_step, 'predict'):
            raise ValueError(f"Final step '{final_name}' must implement predict().")
        return final_step.predict(Xt)

    def predict_proba(self, X) -> np.ndarray:
        """Transform X then return class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        Xt = self.transform(X)
        final_name, final_step = self._final_step()
        if not hasattr(final_step, 'predict_proba'):
            raise ValueError(
                f"Final step '{final_name}' must implement predict_proba()."
            )
        return final_step.predict_proba(Xt)
