import numpy as np
import pytest

from numcompute.optim import grad, jacobian


def test_grad_quadratic():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([3.0, 4.0])
    assert np.allclose(grad(f, x), np.array([6.0, 8.0]), atol=1e-4)


def test_grad_forward():
    def f(x):
        return x[0] ** 2 + 3 * x[1]

    x = np.array([2.0, 5.0])
    assert np.allclose(grad(f, x, method="forward"), np.array([4.0, 3.0]), atol=1e-4)


def test_jacobian_vector_function():
    def F(x):
        return np.array([x[0] + x[1], x[0] * x[1]])

    x = np.array([3.0, 4.0])
    J = jacobian(F, x)
    expected = np.array([[1.0, 1.0], [4.0, 3.0]])
    assert np.allclose(J, expected, atol=1e-4)


def test_grad_invalid_method():
    with pytest.raises(ValueError):
        grad(lambda x: x[0], np.array([1.0]), method="bad")

def test_grad_non_callable_raises():
    with pytest.raises(TypeError):
        grad("not_callable", np.array([1.0]))

def test_grad_invalid_h_raises():
    with pytest.raises(ValueError):
        grad(lambda x: x[0], np.array([1.0]), h=0)

def test_jacobian_scalar_output():
    J = jacobian(lambda x: x[0] ** 2, np.array([3.0]))
    assert np.allclose(J, np.array([[6.0]]), atol=1e-4)        
