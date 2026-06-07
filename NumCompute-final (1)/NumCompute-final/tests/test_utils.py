import numpy as np
import pytest

from numcompute.utils import (
    check_array,
    cosine_similarity,
    euclidean_distance,
    logsumexp,
    make_batches,
    manhattan_distance,
    pairwise_euclidean,
    relu,
    sigmoid,
    stable_softmax,
    topk_indices,
    topk_values,
)


def test_check_array_2d():
    arr = check_array([[1, 2], [3, 4]], ensure_2d=True)
    assert arr.shape == (2, 2)


def test_check_array_1d():
    arr = check_array([1, 2, 3], ensure_1d=True)
    assert arr.shape == (3,)


def test_check_array_no_nan_raises():
    with pytest.raises(ValueError):
        check_array([1.0, np.nan], allow_nan=False)


def test_check_array_no_inf_raises():
    with pytest.raises(ValueError):
        check_array([1.0, np.inf], allow_inf=False)


def test_check_array_copy():
    x = np.array([1.0, 2.0])
    out = check_array(x, copy=True)
    out[0] = 99.0
    assert x[0] == 1.0


def test_check_array_conflicting_dimension_requirements_raises():
    with pytest.raises(ValueError):
        check_array([[1, 2]], ensure_1d=True, ensure_2d=True)


def test_sigmoid_values():
    x = np.array([-1000.0, 0.0, 1000.0])
    out = sigmoid(x)

    assert np.allclose(out[1], 0.5)
    assert out[0] < 1e-10
    assert out[2] > 1 - 1e-10


def test_relu_values():
    x = np.array([-1.0, 0.0, 2.0])
    assert np.array_equal(relu(x), np.array([0.0, 0.0, 2.0]))


def test_logsumexp_matches_manual():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(logsumexp(x), np.log(np.sum(np.exp(x))))


def test_logsumexp_axis_keepdims():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = logsumexp(x, axis=1, keepdims=True)

    assert out.shape == (2, 1)


def test_logsumexp_handles_large_values():
    x = np.array([1000.0, 1000.0])
    out = logsumexp(x)

    assert np.isfinite(out)
    assert np.isclose(out, 1000.0 + np.log(2.0))


def test_logsumexp_all_negative_infinity():
    x = np.array([-np.inf, -np.inf])
    assert logsumexp(x) == -np.inf


def test_stable_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0], [1000.0, 1001.0]])
    out = stable_softmax(x, axis=1)

    assert np.allclose(np.sum(out, axis=1), 1.0)


def test_stable_softmax_handles_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])
    out = stable_softmax(x)

    assert np.all(np.isfinite(out))
    assert np.isclose(out.sum(), 1.0)


def test_euclidean_distance_basic():
    assert np.isclose(euclidean_distance([0, 0], [3, 4]), 5.0)


def test_euclidean_distance_shape_mismatch_raises():
    with pytest.raises(ValueError):
        euclidean_distance([1, 2], [1, 2, 3])


def test_manhattan_distance_basic():
    assert manhattan_distance([1, 2, 3], [3, 2, 1]) == 4.0


def test_cosine_similarity_basic():
    assert np.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert np.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_pairwise_euclidean_shape_and_diagonal():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    D = pairwise_euclidean(X)

    assert D.shape == (2, 2)
    assert np.allclose(np.diag(D), 0.0)


def test_pairwise_euclidean_with_y():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Y = np.array([[0.0, 1.0]])
    D = pairwise_euclidean(X, Y)

    assert D.shape == (2, 1)
    assert np.allclose(D[:, 0], np.array([1.0, np.sqrt(2.0)]))


def test_pairwise_euclidean_feature_mismatch_raises():
    with pytest.raises(ValueError):
        pairwise_euclidean(np.ones((2, 2)), np.ones((3, 3)))


def test_topk_indices_largest():
    x = np.array([1, 9, 3, 7])
    idx = topk_indices(x, 2)

    assert np.array_equal(idx, np.array([1, 3]))


def test_topk_indices_smallest():
    x = np.array([1, 9, 3, 7])
    idx = topk_indices(x, 2, largest=False)

    assert np.array_equal(idx, np.array([0, 2]))


def test_topk_values_largest():
    x = np.array([1, 9, 3, 7])
    vals = topk_values(x, 2)

    assert np.array_equal(vals, np.array([9, 7]))


def test_topk_invalid_k_raises():
    with pytest.raises(ValueError):
        topk_indices(np.array([1, 2, 3]), 0)

    with pytest.raises(ValueError):
        topk_indices(np.array([1, 2, 3]), 4)


def test_make_batches():
    X = np.arange(10)
    batches = list(make_batches(X, 4))

    assert len(batches) == 3
    assert np.array_equal(batches[0], np.array([0, 1, 2, 3]))
    assert np.array_equal(batches[-1], np.array([8, 9]))


def test_make_batches_drop_last():
    X = np.arange(10)
    batches = list(make_batches(X, 4, drop_last=True))

    assert len(batches) == 2
    assert np.array_equal(batches[-1], np.array([4, 5, 6, 7]))


def test_make_batches_shuffle_reproducible():
    X = np.arange(10)
    batches1 = list(make_batches(X, 4, shuffle=True, random_state=42))
    batches2 = list(make_batches(X, 4, shuffle=True, random_state=42))

    assert all(np.array_equal(a, b) for a, b in zip(batches1, batches2))


def test_make_batches_invalid_batch_size_raises():
    with pytest.raises(ValueError):
        list(make_batches(np.arange(10), 0))
