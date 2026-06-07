"""Finite-difference optimization helpers."""

from __future__ import annotations

import numpy as np


def _validate_point(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    return x


def _validate_method(method):
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar-valued function.

    Parameters
    ----------
    f : callable
        Function mapping shape (n,) to scalar.
    x : array-like of shape (n,)
        Evaluation point.
    h : float, default=1e-5
        Finite-difference step size.
    method : {'central', 'forward'}, default='central'
        Finite-difference scheme.

    Returns
    -------
    np.ndarray
        Gradient vector of shape (n,).
    """
    if not callable(f):
        raise TypeError("f must be callable.")
    x = _validate_point(x)
    if h <= 0:
        raise ValueError("h must be positive.")
    _validate_method(method)

    g = np.empty_like(x, dtype=float)
    eye = np.eye(x.size, dtype=float)

    if method == "forward":
        fx = float(f(x))
        for i in range(x.size):
            g[i] = (float(f(x + h * eye[i])) - fx) / h
    else:
        for i in range(x.size):
            g[i] = (float(f(x + h * eye[i])) - float(f(x - h * eye[i]))) / (2.0 * h)
    return g


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian of a vector-valued function.

    Parameters
    ----------
    F : callable
        Function mapping shape (n,) to shape (m,).
    x : array-like of shape (n,)
        Evaluation point.
    h : float, default=1e-5
        Finite-difference step size.
    method : {'central', 'forward'}, default='central'
        Finite-difference scheme.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (m, n).
    """
    if not callable(F):
        raise TypeError("F must be callable.")
    x = _validate_point(x)
    if h <= 0:
        raise ValueError("h must be positive.")
    _validate_method(method)

    Fx = np.asarray(F(x), dtype=float)
    if Fx.ndim == 0:
        Fx = Fx.reshape(1)
    elif Fx.ndim != 1:
        raise ValueError("F(x) must be a 1D array or scalar.")

    J = np.empty((Fx.size, x.size), dtype=float)
    eye = np.eye(x.size, dtype=float)

    if method == "forward":
        for i in range(x.size):
            J[:, i] = (np.asarray(F(x + h * eye[i]), dtype=float) - Fx) / h
    else:
        for i in range(x.size):
            J[:, i] = (
                np.asarray(F(x + h * eye[i]), dtype=float)
                - np.asarray(F(x - h * eye[i]), dtype=float)
            ) / (2.0 * h)
    return J
