"""Неравномерный шаг по времени, возраст во времени, веса и пропуски в варианте со свежестью."""

import numpy as np
import pytest
from conftest import dense_divided_difference, irregular_times, reference_general

from whittaker_trosna import (
    difference_matrix,
    normalized_age_from_times,
    penalty_matrix,
    recency_coefficients,
    recency_whittaker,
    smooth_with_mobility,
    whittaker,
)
from whittaker_trosna.core import _sub_bands, band_matvec, difference_bands, penalty_bands


@pytest.mark.parametrize("order", [1, 2, 3])
def test_divided_differences_match_dense_and_plain(rng, order):
    t = irregular_times(rng, 30)
    np.testing.assert_allclose(difference_matrix(30, order, t).toarray(), dense_divided_difference(t, order))
    # единичный шаг — обычные разности, шаг h — делённые на h^d
    np.testing.assert_allclose(
        difference_matrix(20, order, np.arange(20.0)).toarray(), difference_matrix(20, order).toarray()
    )
    np.testing.assert_allclose(
        difference_matrix(20, order, 0.5 * np.arange(20.0)).toarray(),
        difference_matrix(20, order).toarray() / 0.5**order,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_divided_differences_are_derivatives(rng, order):
    """D_d уничтожает полиномы степени < d и даёт d! на t^d — это d-я производная."""
    t = irregular_times(rng, 40) / 10
    d = difference_matrix(40, order, t)
    for k in range(order):
        np.testing.assert_allclose(d @ t**k, 0.0, atol=1e-8)
    np.testing.assert_allclose(d @ t**order, float(np.prod(range(1, order + 1))), rtol=1e-8)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_irregular_classic_and_recency_match_reference(rng, order):
    n = 50
    t = irregular_times(rng, n)
    y = np.sin(t / 15) + 0.2 * rng.normal(size=n)
    p = dense_divided_difference(t, order)
    p = p.T @ p
    np.testing.assert_allclose(
        whittaker(y, 30.0, t=t, order=order),
        reference_general(y, 30.0, np.ones(n), np.zeros(n, bool), p),
        atol=1e-8,
    )
    c = recency_coefficients(n, "cosine", t=t)
    f = np.zeros(n)
    f[c > 0] = 1 / c[c > 0]
    np.testing.assert_allclose(
        recency_whittaker(y, 30.0, t=t, order=order, weight="cosine"),
        reference_general(y, 30.0, f, c == 0, p),
        atol=1e-8,
    )


def test_trend_in_real_time_is_preserved(rng):
    t = irregular_times(rng, 80)
    y = 3.0 + 0.7 * t
    np.testing.assert_allclose(recency_whittaker(y, 1e6, t=t), y, atol=1e-6)
    # без t нерегулярный тренд по номерам точек — не прямая, и он искажается
    assert np.max(np.abs(recency_whittaker(y, 1e6) - y)) > 1.0


def test_age_is_measured_in_time():
    t = np.array([0.0, 1.0, 2.0, 90.0, 99.0, 100.0])
    np.testing.assert_allclose(normalized_age_from_times(t, 6), [1.0, 0.99, 0.98, 0.10, 0.01, 0.0])
    np.testing.assert_allclose(normalized_age_from_times(t, 6, horizon=10.0), [1, 1, 1, 1, 0.1, 0])
    np.testing.assert_allclose(normalized_age_from_times(None, 3), [1.0, 0.5, 0.0])
    c = recency_coefficients(6, "linear", t=t, horizon=10.0)
    np.testing.assert_allclose(c, [1, 1, 1, 1, 0.1, 0])


@pytest.mark.parametrize(
    ("t", "horizon", "match"),
    [
        ([0.0, 1.0, 1.0], None, "возрастать"),
        ([0.0, 2.0, 1.0], None, "возрастать"),
        ([0.0, np.nan, 1.0], None, "NaN"),
        ([0.0, 1.0], None, "моментов"),
        ([0.0, 1.0, 2.0], 0.0, "горизонт"),
        ([0.0, 1.0, 2.0], -5.0, "горизонт"),
    ],
)
def test_invalid_times(t, horizon, match):
    with pytest.raises(ValueError, match=match):
        recency_whittaker([1.0, 2.0, 3.0], 1.0, t=t, horizon=horizon)


def test_recency_with_missing_values_matches_reference(rng):
    n = 60
    y = np.sin(np.linspace(0, 5, n)) + 0.1 * rng.normal(size=n)
    w = np.ones(n)
    w[[10, 11, 12, 40]] = 0.0
    y_bad = y.copy()
    y_bad[w == 0] = 1e6
    c = recency_coefficients(n, "linear")
    f = np.zeros(n)
    f[c > 0] = w[c > 0] / c[c > 0]
    p = penalty_matrix(n, 2).toarray()
    z = recency_whittaker(y_bad, 20.0, weights=w)
    np.testing.assert_allclose(z, reference_general(y, 20.0, f, (c == 0) & (w > 0), p), atol=1e-8)
    assert np.all(np.abs(z[[10, 11, 12, 40]]) < 1.5)


def test_missing_newest_point_is_extrapolated():
    """Если пропущена самая свежая точка, закреплять нечего — её значение экстраполируется."""
    y = np.arange(10.0)
    w = np.ones(10)
    w[-1] = 0.0
    y_bad = y.copy()
    y_bad[-1] = -999.0
    z = recency_whittaker(y_bad, 100.0, weights=w)
    assert z[-1] == pytest.approx(9.0, abs=1e-8)


def test_weights_scale_fidelity(rng):
    """Вес w умножает верность: f = w / c (эквивалентно c / w)."""
    y = rng.normal(size=30)
    c = recency_coefficients(30, "linear")
    w = rng.uniform(0.2, 3.0, size=30)
    z1 = smooth_with_mobility(y, 10.0, c, weights=w)
    p = penalty_matrix(30, 2).toarray()
    f = np.zeros(30)
    f[c > 0] = w[c > 0] / c[c > 0]
    np.testing.assert_allclose(z1, reference_general(y, 10.0, f, c == 0, p), atol=1e-9)


def test_singular_system_is_reported():
    with pytest.raises(ValueError, match="вырождена"):
        recency_whittaker([1.0, 2.0, 3.0, 4.0], 1.0, weights=[0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 3, 6, 40])
@pytest.mark.parametrize("irregular", [False, True])
def test_band_representation_matches_sparse(rng, order, n, irregular):
    t = irregular_times(rng, n) if irregular else None
    dense_p = penalty_matrix(n, order, t).toarray()
    pb = penalty_bands(difference_bands(n, order, t), n)
    for m in range(min(order + 1, n)):
        np.testing.assert_allclose(pb[m, : n - m], np.diag(dense_p, m), atol=1e-12)
    x = rng.normal(size=n)
    np.testing.assert_allclose(band_matvec(pb, x), dense_p @ x, atol=1e-10)
    idx = np.sort(rng.choice(n, size=max(1, n // 2), replace=False))
    sub = _sub_bands(pb, idx)
    dense_sub = dense_p[np.ix_(idx, idx)]
    for m in range(min(order + 1, idx.size)):
        np.testing.assert_allclose(sub[m, : idx.size - m], np.diag(dense_sub, m), atol=1e-12)
    # вне ленты подматрица действительно нулевая
    assert np.allclose(np.triu(dense_sub, order + 1), 0.0)
