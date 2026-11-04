"""Fast checks (~2s): pytest experiments/01-turboquant-kv-cache"""
import numpy as np
import pytest

from turboquant import (TurboQuantMSE, TurboQuantProd, UniformMinMax,
                        mse_lower_bound, mse_upper_bound)

D, N = 128, 3000


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, D))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_mse_within_paper_bounds(X, bits):
    mse = np.mean(np.sum((X - TurboQuantMSE(D, bits)(X)) ** 2, axis=1))
    assert mse_lower_bound(bits) < mse < mse_upper_bound(bits)


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_prod_is_unbiased(X, bits):
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((N, D))
    true = np.sum(X * Y, axis=1)
    est = np.sum(TurboQuantProd(D, bits)(X) * Y, axis=1)
    assert abs(np.dot(est, true) / np.dot(true, true) - 1) < 0.03


def test_mse_one_bit_shrinks_by_two_over_pi(X):
    est = np.sum(TurboQuantMSE(D, 1)(X) * X, axis=1)
    assert abs(est.mean() - 2 / np.pi) < 0.02


def test_scale_is_preserved():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, D)) * 37.0
    Xh = TurboQuantMSE(D, 4)(X)
    assert np.mean(np.sum((X - Xh) ** 2, 1) / np.sum(X ** 2, 1)) < mse_upper_bound(4)


def test_bit_accounting():
    assert TurboQuantMSE(D, 3).bits_per_coord() == 3 + 16 / D
    assert UniformMinMax(D, 3).bits_per_coord() == 3 + 32 / D
