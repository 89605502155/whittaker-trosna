"""Блочная перекрёстная проверка ("bcv") для коррелированного шума."""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import irregular_times

from whittaker_trosna import (
    StreamingWhittaker,
    fit_recency_whittaker,
    recency_coefficients,
    recency_whittaker,
    select_lambda,
    smooth_with_mobility,
    whittaker,
)
from whittaker_trosna.core import _Factor, _solve, build_problem, solve_problem
from whittaker_trosna.lambda_selection import block_cv_score, cv_score, default_cv_block


def ar1_noise(rng, n, phi, sd=1.0):
    e = np.empty(n)
    e[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0.0, sd * np.sqrt(1.0 - phi**2))
    return e


def memory_signal(n):
    t = np.arange(n) * 2.0
    return 3 * np.sin(2 * np.pi * t / 900) + 1.5 * np.sin(2 * np.pi * t / 240) + 0.002 * t + 2.0 * (t > 1500)


@pytest.mark.parametrize("bandwidth", [1, 2, 3])
@pytest.mark.parametrize(("m", "width"), [(1, 4), (2, 1), (7, 3), (40, 1), (40, 9), (40, 60)])
def test_inverse_band_matches_dense(rng, bandwidth, m, width):
    d = np.diff(np.eye(m + bandwidth), bandwidth, axis=0)[:, :m] if m > 1 else np.zeros((1, 1))
    a = np.diag(rng.uniform(0.5, 3.0, size=m)) + 7.0 * d.T @ d
    factor = _Factor.from_matrix(sp.csr_matrix(a), bandwidth)
    band = factor.inverse_band(width)
    inv = np.linalg.inv(a)
    width = band.shape[1] - 1
    expected = np.array([[inv[i, i + j] if i + j < m else 0.0 for j in range(width + 1)] for i in range(m)])
    np.testing.assert_allclose(band, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(band[:, 0], factor.inverse_diagonal(), rtol=1e-12)


def brute_block_cv(y, lam, block, *, t=None, order=2, weights=None, mobility=None):
    """Честное исключение блока: вес 0 у точки и у ``block`` соседних свободных точек с каждой стороны."""
    n = y.size
    w0 = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    c = np.ones(n) if mobility is None else np.asarray(mobility, dtype=float)
    free = np.flatnonzero(~((c == 0) & (w0 > 0)))
    errors = []
    for pos, i in enumerate(free):
        if w0[i] == 0:
            continue
        w = w0.copy()
        w[free[max(0, pos - block) : pos + block + 1]] = 0.0
        problem = build_problem(y, t=t, order=order, weights=w, mobility=c)
        errors.append(y[i] - solve_problem(problem, lam)[i])
    return np.sqrt(np.mean(np.square(errors)))


@pytest.mark.parametrize("block", [1, 3, 6])
@pytest.mark.parametrize("weight", [None, "linear", "power"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_block_cv_equals_brute_force(rng, block, weight, order):
    """(I - H_BB)⁻¹ r_B совпадает с честным пересчётом без блока (время, пропуски, свежесть)."""
    n = 45
    t = irregular_times(rng, n)
    y = np.sin(t / 15.0) + 0.3 * rng.normal(size=n)
    w = rng.uniform(0.3, 2.0, size=n)
    w[[5, 6, 20]] = 0.0
    c = None if weight is None else recency_coefficients(n, weight, t=t)
    lam = 40.0
    problem = build_problem(y, t=t, order=order, weights=w, mobility=c)
    expected = brute_block_cv(y, lam, block, t=t, order=order, weights=w, mobility=c)
    assert block_cv_score(problem, lam, block) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(("n", "block"), [(10, 1), (40, 2), (100, 5), (150, 7), (200, 10), (5000, 10)])
def test_default_block_scales_with_length(rng, n, block):
    assert default_cv_block(n) == block
    problem = build_problem(rng.normal(size=n))
    assert block_cv_score(problem, 30.0) == pytest.approx(block_cv_score(problem, 30.0, block), rel=1e-12)


def test_short_series_bcv_does_not_oversmooth():
    """На 100 точках (типичная длина для модели) "bcv" по умолчанию не хуже "cv" на белом шуме
    и лучше на коррелированном. На 30 рядах медианное отношение к наилучшей ошибке у "bcv" — 1,08 (белый
    шум) и 1,13 (φ = 0,6), у "cv" — 1,03 и 1,45."""
    n = 100
    t = np.arange(n) * 2.0
    truth = 3 * np.sin(2 * np.pi * t / 150) + 1.5 * np.sin(2 * np.pi * t / 60) + 0.01 * t
    lams = np.logspace(-2, 9, 45)
    for phi, limit in ((0.0, 1.35), (0.6, 1.35)):
        ratios = []
        for seed in range(10):
            y = truth + ar1_noise(np.random.default_rng(seed), n, phi)
            best = min(np.sqrt(np.mean((whittaker(y, lam) - truth) ** 2)) for lam in lams)
            ratios.append(np.sqrt(np.mean((whittaker(y, "bcv") - truth) ** 2)) / best)
        assert np.median(ratios) < limit


def test_zero_block_is_ordinary_cv(rng):
    y = rng.normal(size=60)
    problem = build_problem(y, mobility=recency_coefficients(60, "linear"))
    assert block_cv_score(problem, 25.0, 0) == pytest.approx(cv_score(problem, 25.0), rel=1e-10)


def test_block_is_clipped_for_short_series(rng):
    y = rng.normal(size=12)
    score = block_cv_score(build_problem(y), 10.0, 50)
    assert np.isfinite(score)


def test_correlated_noise_fools_cv_but_not_bcv():
    """На шуме AR(1) обычная CV сильно недосглаживает, блочная выбирает λ, близкую к наилучшей.

    На 8 зёрнах при φ = 0,6 отношение ошибки к наилучшей: у "cv" в среднем 2,56, у "bcv" — 1,06;
    наихудший случай "bcv" — 1,23 (у break_serial_correlation из whittaker-eilers в среднем 1,95).
    """
    n = 800
    truth = memory_signal(n)
    lams = np.logspace(-1, 9, 41)
    ratios = {"cv": [], "bcv": []}
    for seed in range(3):
        y = truth + ar1_noise(np.random.default_rng(seed), n, 0.6)
        errors = np.array([np.sqrt(np.mean((whittaker(y, lam) - truth) ** 2)) for lam in lams])
        best = errors.min()
        for method, values in ratios.items():
            z = whittaker(y, method)
            values.append(np.sqrt(np.mean((z - truth) ** 2)) / best)
    assert np.mean(ratios["bcv"]) < 1.25
    assert np.mean(ratios["cv"]) > 1.6
    assert max(ratios["bcv"]) < 1.4


def test_bcv_with_recency_on_correlated_noise():
    n = 800
    truth = memory_signal(n)
    y = truth + ar1_noise(np.random.default_rng(11), n, 0.6)
    lam_cv = select_lambda(y, "cv", weight="linear").lam
    lam_bcv = select_lambda(y, "bcv", weight="linear").lam
    assert lam_bcv > 10 * lam_cv

    def err(lam):
        return np.sqrt(np.mean((recency_whittaker(y, lam) - truth) ** 2))

    assert err(lam_bcv) < 0.7 * err(lam_cv)


def test_bcv_on_white_noise_agrees_with_cv(rng):
    """Без корреляции блочная проверка не портит выбор."""
    n = 600
    truth = memory_signal(n)
    y = truth + rng.normal(size=n)
    lam_cv = select_lambda(y, "cv").lam
    lam_bcv = select_lambda(y, "bcv").lam
    assert 0.1 < lam_bcv / lam_cv < 10.0


def test_cv_block_reaches_every_entry_point(rng):
    n = 200
    t = irregular_times(rng, n)
    y = memory_signal(n) + ar1_noise(rng, n, 0.5)
    problem = build_problem(y, t=t)
    search = select_lambda(y, "bcv", t=t, cv_block=4)
    assert search.method == "bcv"
    assert block_cv_score(problem, search.lam, 4) == pytest.approx(search.score, rel=1e-12)
    z_classic = whittaker(y, "bcv", t=t, cv_block=4)
    np.testing.assert_allclose(z_classic, solve_problem(problem, search.lam), rtol=1e-12, atol=1e-12)

    res = fit_recency_whittaker(y, "bcv", t=t, cv_block=4, strength=2.0)
    assert res.search.method == "bcv"
    assert res.lam == pytest.approx(2.0 * res.search.lam)
    np.testing.assert_allclose(recency_whittaker(y, "bcv", t=t, cv_block=4, strength=2.0), res.z)
    z_core = smooth_with_mobility(y, "bcv", res.mobility, t=t, cv_block=4, strength=2.0)
    np.testing.assert_allclose(z_core, res.z)


def test_streaming_bcv(rng):
    n = 300
    y = memory_signal(n) + ar1_noise(rng, n, 0.6)
    s = StreamingWhittaker("bcv", cv_block=5, retune_every=1000)
    s.extend(y, np.arange(n) * 2.0)
    snap = s.smooth()
    expected = fit_recency_whittaker(y, "bcv", t=np.arange(n) * 2.0, cv_block=5)
    assert snap.search.method == "bcv"
    assert snap.lam == pytest.approx(expected.lam, rel=1e-10)
    np.testing.assert_allclose(snap.z, expected.z, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("lam", "cv_block", "match"),
    [
        ("cv", 5, "только к method='bcv'"),
        (10.0, 5, "только при подборе"),
        ("bcv", 0, "целым ≥ 1"),
        ("bcv", 2.5, "целым ≥ 1"),
        ("bcv", True, "целым ≥ 1"),
    ],
)
def test_invalid_cv_block(lam, cv_block, match):
    y = np.sin(np.linspace(0, 5, 50))
    with pytest.raises(ValueError, match=match):
        recency_whittaker(y, lam, cv_block=cv_block)
    with pytest.raises(ValueError, match=match):
        StreamingWhittaker(lam, cv_block=cv_block)


def test_bcv_score_respects_pinned_points(rng):
    """Закреплённая последняя точка не участвует в оценке и не исключается."""
    n = 50
    y = rng.normal(size=n)
    c = recency_coefficients(n, "linear")
    problem = build_problem(y, mobility=c)
    z, _ = _solve(problem, 30.0)
    assert z[-1] == y[-1]
    assert np.isfinite(block_cv_score(problem, 30.0, 3))
