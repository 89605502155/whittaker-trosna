"""Автоподбор λ, сила сглаживания, частота среза."""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import irregular_times

from whittaker_trosna import (
    cutoff_for_lambda,
    fit_recency_whittaker,
    lambda_for_cutoff,
    recency_coefficients,
    recency_whittaker,
    select_lambda,
    whittaker,
)
from whittaker_trosna.core import SmoothingProblem, _Factor, _solve, build_problem, solve_problem
from whittaker_trosna.lambda_selection import cv_score, forward_score, gcv_score, hat_diagonal


@pytest.mark.parametrize("bandwidth", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [1, 2, 5, 40])
def test_inverse_diagonal_of_banded_matrix(rng, bandwidth, m):
    d = np.diff(np.eye(m + bandwidth), bandwidth, axis=0)[:, :m] if m > 1 else np.zeros((1, 1))
    a = np.diag(rng.uniform(0.5, 3.0, size=m)) + 7.0 * d.T @ d
    factor = _Factor.from_matrix(sp.csr_matrix(a), bandwidth)
    np.testing.assert_allclose(factor.inverse_diagonal(), np.diag(np.linalg.inv(a)), rtol=1e-10)


def smoother_matrix(problem, lam):
    cols = []
    for e in np.eye(problem.n):
        sub = SmoothingProblem(e, problem.w, problem.c, t=problem.t, order=problem.order)
        cols.append(_solve(sub, lam)[0])
    return np.column_stack(cols)


@pytest.mark.parametrize("weight", [None, "linear", "power"])
def test_hat_diagonal_matches_dense_smoother_matrix(rng, weight):
    n = 35
    t = irregular_times(rng, n)
    y = rng.normal(size=n)
    c = None if weight is None else recency_coefficients(n, weight, t=t)
    problem = build_problem(y, t=t, mobility=c)
    _, h = hat_diagonal(problem, 40.0)
    np.testing.assert_allclose(h, np.diag(smoother_matrix(problem, 40.0)), atol=1e-10)


@pytest.mark.parametrize("weight", [None, "linear", "cosine"])
def test_cv_equals_brute_force_leave_one_out(rng, weight):
    """Формула (y - z)/(1 - h) совпадает с честным исключением каждой точки."""
    n = 40
    y = np.sin(np.linspace(0, 4, n)) + 0.3 * rng.normal(size=n)
    c = np.ones(n) if weight is None else recency_coefficients(n, weight)
    lam = 15.0
    errors = []
    for i in range(n):
        if c[i] == 0:
            continue
        w = np.ones(n)
        w[i] = 0.0
        z = build_problem(y, weights=w, mobility=c)
        errors.append(y[i] - solve_problem(z, lam)[i])
    brute = np.sqrt(np.mean(np.square(errors)))
    assert cv_score(build_problem(y, mobility=c), lam) == pytest.approx(brute, rel=1e-9)


def test_gcv_formula(rng):
    n = 30
    y = rng.normal(size=n)
    problem = build_problem(y)
    h = smoother_matrix(problem, 5.0)
    r = y - h @ y
    expected = n * np.sum(r**2) / (n - np.trace(h)) ** 2
    assert gcv_score(problem, 5.0) == pytest.approx(expected, rel=1e-9)


def test_forward_score_is_honest_forecast(rng):
    """Прогноз в t_i строится только по прошлому, причём свежесть пересчитана для прошлого."""
    n = 30
    t = irregular_times(rng, n)
    y = np.sin(t / 10) + 0.1 * rng.normal(size=n)
    rule = lambda tt, k: recency_coefficients(k, "linear", t=tt)  # noqa: E731
    problem = build_problem(y, t=t, rule=rule)
    i = 25
    sub = problem.prefix_with_target(i)
    # точка i имеет вес 0, будущих точек нет, последняя точка прошлого закреплена
    assert sub.n == i + 1
    assert sub.w[-1] == 0.0
    assert sub.fixed[i - 1]
    assert not sub.fixed[-1]
    np.testing.assert_array_equal(sub.y[:i], y[:i])
    pred = forward_score([(sub, 0.0)], 50.0)  # «цель» 0 → score = |прогноз|
    # замена будущих значений не влияет на прогноз
    y2 = y.copy()
    y2[i:] = 1e6
    sub2 = build_problem(y2, t=t, rule=rule).prefix_with_target(i)
    assert forward_score([(sub2, 0.0)], 50.0) == pytest.approx(pred, rel=1e-12)


@pytest.mark.parametrize(
    ("method", "weight"),
    [
        ("cv", None),
        ("gcv", None),
        ("forward", None),
        ("bcv", None),
        ("cv", "linear"),
        ("gcv", "linear"),
        ("bcv", "linear"),
    ],
)
def test_selected_lambda_is_sensible(rng, method, weight):
    n = 300
    truth = np.sin(np.linspace(0, 6 * np.pi, n))
    y = truth + 0.3 * rng.normal(size=n)
    search = select_lambda(y, method, weight=weight)
    assert np.isfinite(search.lam)
    assert search.lam > 0
    # минимум внутри сетки, а не на краю
    assert search.lams[0] < search.lam < search.lams[-1]

    def err(lam):
        z = whittaker(y, lam) if weight is None else recency_whittaker(y, lam, weight=weight)
        return np.sqrt(np.mean((z - truth)[: n // 2] ** 2))

    assert err(search.lam) < err(search.lam / 1000)
    assert err(search.lam) < err(search.lam * 1000)


def test_forward_with_pinned_newest_point_warns(rng):
    y = np.sin(np.linspace(0, 10, 120)) + 0.3 * rng.normal(size=120)
    with pytest.warns(UserWarning, match="малоинформативна"):
        select_lambda(y, "forward", weight="linear")


def test_auto_lambda_beats_raw_and_matches_oracle(rng):
    """λ из CV даёт ошибку не более чем на 30 % хуже наилучшей λ (оракула).

    По 8 случайным реализациям среднее отношение ≈ 1.07, худшее ≈ 1.3.
    """
    n = 400
    truth = np.sin(np.linspace(0, 4 * np.pi, n)) + 0.5 * np.cos(np.linspace(0, 11, n))
    y = truth + 0.4 * rng.normal(size=n)
    grid = np.logspace(-2, 8, 81)
    errs = [np.sqrt(np.mean((whittaker(y, g) - truth) ** 2)) for g in grid]
    z = whittaker(y, "cv")
    assert np.sqrt(np.mean((z - truth) ** 2)) < 1.3 * min(errs)
    assert np.sqrt(np.mean((z - truth) ** 2)) < 0.5 * np.sqrt(np.mean((y - truth) ** 2))


def test_strength_multiplies_lambda(rng):
    y = rng.normal(size=100).cumsum()
    base = fit_recency_whittaker(y, "gcv")
    strong = fit_recency_whittaker(y, "gcv", strength=10.0)
    assert strong.lam == pytest.approx(10.0 * base.lam)
    assert strong.search.lam == pytest.approx(base.search.lam)
    np.testing.assert_allclose(strong.z, recency_whittaker(y, 10.0 * base.lam), atol=1e-10)
    assert fit_recency_whittaker(y, 7.0, strength=0.5).lam == 3.5
    # сильнее сглаживание — меньше шероховатость
    assert np.sum(np.diff(strong.z, 2) ** 2) < np.sum(np.diff(base.z, 2) ** 2)
    with pytest.raises(ValueError, match="сила"):
        recency_whittaker(y, 1.0, strength=0.0)


def test_auto_lambda_with_irregular_times_and_missing(rng):
    n = 200
    t = irregular_times(rng, n)
    truth = np.sin(t / 40)
    y = truth + 0.2 * rng.normal(size=n)
    w = np.ones(n)
    w[50:60] = 0.0
    res = fit_recency_whittaker(y, "cv", t=t, weights=w)
    assert np.sqrt(np.mean((res.z - truth)[:150] ** 2)) < 0.12
    assert res.z[-1] == y[-1]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_cutoff_period_halves_amplitude(order):
    """При λ = lambda_for_cutoff(P) синусоида с периодом P ослабляется вдвое."""
    n = 4000
    x = np.arange(n, dtype=float)
    for period, check in [
        (40.0, lambda g: abs(g - 0.5) < 0.01),
        (400.0, lambda g: g > 0.97),
        (8.0, lambda g: g < 0.05),
    ]:
        lam = lambda_for_cutoff(40.0, order, dt=1.0)
        z = whittaker(np.sin(2 * np.pi * x / period), lam, order=order)
        mid = slice(n // 4, 3 * n // 4)
        gain = np.max(np.abs(z[mid])) / 1.0
        assert check(gain), (period, gain)


def test_cutoff_helpers():
    assert lambda_for_cutoff(2 * np.pi, 2) == pytest.approx(1.0)
    assert cutoff_for_lambda(lambda_for_cutoff(123.0, 3), 3) == pytest.approx(123.0)
    # при большом периоде точная формула сходится к приближённой
    assert lambda_for_cutoff(1000.0, 2, dt=1.0) == pytest.approx(lambda_for_cutoff(1000.0, 2), rel=1e-4)
    with pytest.raises(ValueError, match="двух шагов"):
        lambda_for_cutoff(1.5, 2, dt=1.0)
    with pytest.raises(ValueError, match="период"):
        lambda_for_cutoff(-1.0)


def test_invalid_lambda_selection():
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="способ"):
        recency_whittaker(y, "magic")
    with pytest.raises(ValueError, match="больше"):
        select_lambda([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="диапазон"):
        select_lambda(y, lam_range=(10.0, 1.0))
