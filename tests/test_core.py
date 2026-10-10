"""Математическая корректность: совпадение с эталонами и свойства оптимума."""

import time

import numpy as np
import pytest
from conftest import dense_difference, reference_classic, reference_recency

from whittaker_trosna import (
    difference_matrix,
    objective,
    penalty_matrix,
    recency_coefficients,
    recency_whittaker,
    smooth_with_mobility,
    whittaker,
)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 5, 30])
def test_difference_and_penalty_matrices(n, order):
    if n <= order:
        assert difference_matrix(n, order).shape == (0, n)
        return
    np.testing.assert_allclose(difference_matrix(n, order).toarray(), dense_difference(n, order))
    d = dense_difference(n, order)
    np.testing.assert_allclose(penalty_matrix(n, order).toarray(), d.T @ d)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("lam", [0.1, 10.0, 1e4])
def test_classic_matches_dense_reference(rng, order, lam):
    y = rng.normal(size=60).cumsum()
    np.testing.assert_allclose(whittaker(y, lam, order=order), reference_classic(y, lam, order), atol=1e-8)


def test_classic_with_missing_values(rng):
    y = np.sin(np.linspace(0, 6, 80)) + 0.1 * rng.normal(size=80)
    w = np.ones(80)
    w[20:30] = 0.0
    y_bad = y.copy()
    y_bad[20:30] = 1e6  # мусор в пропусках не должен влиять
    z = whittaker(y_bad, 50.0, weights=w)
    np.testing.assert_allclose(z, reference_classic(y, 50.0, 2, w), atol=1e-8)
    assert np.all(np.abs(z[20:30]) < 1.5)


@pytest.mark.parametrize("weight", ["linear", "cosine", "power", "exponential", "logistic"])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("lam", [0.5, 100.0, 1e5])
def test_recency_matches_kkt_reference(rng, weight, order, lam):
    y = rng.normal(size=50).cumsum() + rng.normal(size=50)
    c = recency_coefficients(50, weight)
    z = recency_whittaker(y, lam, order=order, weight=weight)
    np.testing.assert_allclose(z, reference_recency(y, lam, c, order), atol=1e-7, rtol=1e-7)


def test_arbitrary_fixed_points_match_reference(rng):
    y = rng.normal(size=40)
    c = rng.uniform(0.05, 1.0, size=40)
    c[[0, 7, 8, 25, 39]] = 0.0
    z = smooth_with_mobility(y, 30.0, c)
    np.testing.assert_allclose(z, reference_recency(y, 30.0, c, 2), atol=1e-9)
    np.testing.assert_array_equal(z[c == 0], y[c == 0])


def test_unit_mobility_equals_classic(rng):
    y = rng.normal(size=100)
    for order in (1, 2, 3):
        np.testing.assert_allclose(
            smooth_with_mobility(y, 77.0, np.ones(100), order=order),
            whittaker(y, 77.0, order=order),
            atol=1e-10,
        )


@pytest.mark.parametrize("weight", ["linear", "power", "exponential", "cosine", "logistic"])
def test_newest_point_is_never_moved(rng, weight):
    y = rng.normal(size=200) * 10
    for lam in (1e-3, 1.0, 1e3, 1e8):
        z = recency_whittaker(y, lam, weight=weight)
        assert z[-1] == y[-1]


def test_small_mobility_converges_to_fixed(rng):
    y = rng.normal(size=30)
    c = recency_coefficients(30, "linear")
    z_fixed = smooth_with_mobility(y, 10.0, c)
    errors = []
    for eps in (1e-2, 1e-4, 1e-6):
        c_eps = c.copy()
        c_eps[-1] = eps
        errors.append(np.max(np.abs(smooth_with_mobility(y, 10.0, c_eps) - z_fixed)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-5


def test_solution_is_minimum_of_objective(rng):
    y = rng.normal(size=40)
    c = recency_coefficients(40, "cosine")
    z = recency_whittaker(y, 25.0, weight="cosine")
    j0 = objective(y, z, 25.0, c)
    for _ in range(50):
        v = rng.normal(size=40)
        v[c == 0] = 0.0
        assert objective(y, z + 1e-3 * v, 25.0, c) > j0
    # градиент по свободным переменным равен нулю
    d = dense_difference(40, 2)
    free = c > 0
    grad = 2 * (z - y)[free] / c[free] + 2 * 25.0 * (d.T @ d @ z)[free]
    np.testing.assert_allclose(grad, 0.0, atol=1e-9)
    # нарушение закрепления делает функционал бесконечным
    z_bad = z.copy()
    z_bad[-1] += 1.0
    assert objective(y, z_bad, 25.0, c) == np.inf


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polynomials_below_order_are_preserved(order):
    t = np.linspace(-1, 1, 64)
    y = sum((k + 1.5) * t**k for k in range(order))
    for weight in ("linear", "power", "cosine"):
        np.testing.assert_allclose(recency_whittaker(y, 1e6, order=order, weight=weight), y, atol=1e-6)


def test_linearity_and_affine_equivariance(rng):
    y1, y2 = rng.normal(size=(2, 50))
    f = lambda y: recency_whittaker(y, 40.0, weight="exponential", k=3.0)  # noqa: E731
    np.testing.assert_allclose(f(2.5 * y1 - 3.0 * y2), 2.5 * f(y1) - 3.0 * f(y2), atol=1e-10)
    np.testing.assert_allclose(f(4.0 * y1 + 7.0), 4.0 * f(y1) + 7.0, atol=1e-9)


def test_hat_matrix_structure():
    """Матрица сглаживания H (z = H y): последняя строка — орт, строки суммируются в 1."""
    n = 25
    h = np.column_stack([recency_whittaker(e, 50.0) for e in np.eye(n)])
    np.testing.assert_array_equal(h[-1], np.eye(n)[-1])
    np.testing.assert_allclose(h.sum(axis=1), 1.0, atol=1e-10)


def test_lambda_extremes(rng):
    y = rng.normal(size=60)
    np.testing.assert_array_equal(recency_whittaker(y, 0.0), y)
    np.testing.assert_allclose(recency_whittaker(y, 1e-9), y, atol=1e-6)
    # λ → ∞: прямая (order=2), проходящая через закреплённую последнюю точку
    z = recency_whittaker(y, 1e10)
    assert np.max(np.abs(np.diff(z, 2))) < 1e-6
    assert z[-1] == y[-1]


def test_short_series():
    np.testing.assert_array_equal(recency_whittaker([3.0], 10.0), [3.0])
    np.testing.assert_array_equal(recency_whittaker([3.0, 5.0], 10.0), [3.0, 5.0])
    z = recency_whittaker([0.0, 10.0, 0.0], 1e9)  # прямая через последнюю точку
    assert z[-1] == 0.0
    assert abs(z[0] - 2 * z[1] + z[2]) < 1e-6


def test_input_is_not_modified(rng):
    y = rng.normal(size=20)
    y0 = y.copy()
    recency_whittaker(y, 10.0)
    whittaker(y, 10.0)
    np.testing.assert_array_equal(y, y0)


def test_accepts_lists_and_ints():
    z = recency_whittaker([1, 2, 4, 8, 16], 1)
    assert z.dtype == np.float64
    assert z[-1] == 16.0


@pytest.mark.parametrize(
    ("y", "lam", "order", "match"),
    [
        ([], 1.0, 2, "пуст"),
        ([[1.0, 2.0]], 1.0, 2, "одномерный"),
        ([1.0, np.nan, 2.0], 1.0, 2, "NaN"),
        ([1.0, np.inf, 2.0], 1.0, 2, "NaN"),
        ([1.0, 2.0, 3.0], -1.0, 2, "λ"),
        ([1.0, 2.0, 3.0], np.nan, 2, "λ"),
        ([1.0, 2.0, 3.0], 1.0, 0, "порядок"),
        ([1.0, 2.0, 3.0], 1.0, 1.5, "порядок"),
        ([1.0, 2.0, 3.0], 1.0, True, "порядок"),
    ],
)
def test_invalid_inputs(y, lam, order, match):
    with pytest.raises(ValueError, match=match):
        recency_whittaker(y, lam, order=order)
    with pytest.raises(ValueError, match=match):
        whittaker(y, lam, order=order)


def test_classic_invalid_weights():
    with pytest.raises(ValueError, match="неотрицательными"):
        whittaker([1.0, 2.0, 3.0], 1.0, weights=[1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="ожидалось 3 весов"):
        whittaker([1.0, 2.0, 3.0], 1.0, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="вырождена"):
        whittaker([1.0, 2.0, 3.0, 4.0], 1.0, weights=[0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="без штрафа"):
        whittaker([1.0, 2.0, 3.0], 0.0, weights=[1.0, 0.0, 1.0])


def test_large_series_is_fast(rng):
    n = 200_000
    y = rng.normal(size=n).cumsum()
    start = time.perf_counter()
    z = recency_whittaker(y, 1e4, order=3)
    assert time.perf_counter() - start < 5.0
    assert z[-1] == y[-1]
    assert np.all(np.isfinite(z))
