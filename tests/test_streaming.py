"""Потоковый режим: точки дописываются, окно скользит, результат совпадает с пакетным."""

import time

import numpy as np
import pytest
from conftest import irregular_times

from whittaker_trosna import StreamingWhittaker, fit_recency_whittaker, recency_whittaker, whittaker


def memory_trace(rng, n):
    """Похоже на потребление памяти ВМ: тренд, «ступенька» нагрузки и шум; шаг 1–5 с."""
    t = np.cumsum(rng.choice([1.0, 1.0, 2.0, 5.0], size=n))
    level = 2048 + 0.5 * t + 300 * (t > t[n // 2]) + 40 * np.sin(t / 30)
    return t, level + 15 * rng.normal(size=n)


def test_stream_equals_batch_on_window(rng):
    t, y = memory_trace(rng, 300)
    s = StreamingWhittaker(500.0, weight="cosine", window=200.0)
    for v, tt in zip(y, t, strict=True):
        s.append(v, tt)
    snap = s.smooth()
    keep = t >= t[-1] - 200.0
    np.testing.assert_array_equal(snap.t, t[keep])
    np.testing.assert_allclose(
        snap.z, recency_whittaker(y[keep], 500.0, t=t[keep], weight="cosine"), atol=1e-9
    )
    assert snap.z[-1] == y[-1]


def test_stream_equals_batch_at_every_step(rng):
    t = irregular_times(rng, 60)
    y = rng.normal(size=60).cumsum()
    s = StreamingWhittaker(30.0, weight="linear", horizon=20.0, max_points=25)
    for i in range(60):
        s.append(y[i], t[i])
        lo = max(0, i + 1 - 25)
        expected = (
            recency_whittaker(y[lo : i + 1], 30.0, t=t[lo : i + 1], horizon=20.0)
            if i + 1 - lo > 4
            else y[lo : i + 1]
        )
        np.testing.assert_allclose(s.smooth().z, expected, atol=1e-9)


def test_classic_stream_and_strength(rng):
    y = rng.normal(size=100)
    s = StreamingWhittaker(10.0, weight=None, strength=3.0)
    s.extend(y)
    np.testing.assert_allclose(s.smooth().z, whittaker(y, 30.0, t=np.arange(100.0)), atol=1e-9)
    assert s.lam == 30.0


def test_window_trimming():
    s = StreamingWhittaker(1.0, window=10.0, max_points=8)
    s.extend(np.arange(20.0), times=np.arange(20.0) * 0.5)
    assert len(s) == 8  # max_points сильнее окна в 10 с (там было бы 21 точка)
    s2 = StreamingWhittaker(1.0, window=10.0)
    s2.extend(np.arange(30.0), times=np.arange(30.0))
    np.testing.assert_array_equal(s2.smooth().t, np.arange(19.0, 30.0))


def test_auto_lambda_retuning(rng):
    t, y = memory_trace(rng, 400)
    s = StreamingWhittaker("cv", window=600.0, retune_every=50)
    s.extend(y[:200], t[:200])
    first = s.smooth()
    assert first.search is not None
    lam0 = s.lam
    s.extend(y[200:240], t[200:240])
    assert s.smooth().lam == lam0  # 40 < 50 новых точек — без нового подбора
    s.extend(y[240:260], t[240:260])
    snap = s.smooth()
    assert snap.search is not first.search  # подобрали заново
    expected = fit_recency_whittaker(snap.y, "cv", t=snap.t)
    assert s.lam == pytest.approx(expected.lam, rel=1e-6)


def test_missing_values_in_stream(rng):
    t, y = memory_trace(rng, 120)
    y_nan = y.copy()
    y_nan[[30, 31, 70]] = np.nan
    s = StreamingWhittaker(200.0)
    s.extend(y_nan, t)
    snap = s.smooth()
    w = np.isfinite(y_nan).astype(float)
    np.testing.assert_allclose(
        snap.z, recency_whittaker(np.nan_to_num(y_nan), 200.0, t=t, weights=w), atol=1e-8
    )
    assert np.isnan(snap.y[30])
    assert np.isfinite(snap.z[30])
    # пропуск в самой свежей точке — она экстраполируется
    s.append(np.nan, t[-1] + 1)
    assert np.isfinite(s.latest)


def test_caching_and_defaults():
    s = StreamingWhittaker(5.0)
    s.append(1.0)
    s.append(2.0)
    np.testing.assert_array_equal(s.smooth().t, [0.0, 1.0])
    np.testing.assert_array_equal(s.smooth().z, [1.0, 2.0])  # мало точек — как есть
    s.extend([4.0, 3.0, 5.0, 4.5])
    snap = s.smooth()
    assert s.smooth() is snap  # без новых данных — без пересчёта
    s.append(6.0)
    assert s.smooth() is not snap
    assert s.latest == 6.0


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"lam": "magic"}, ValueError),
        ({"lam": -1.0}, ValueError),
        ({"lam": 1.0, "strength": 0.0}, ValueError),
        ({"lam": 1.0, "window": 0.0}, ValueError),
        ({"lam": 1.0, "max_points": 0}, ValueError),
        ({"lam": 1.0, "retune_every": 0}, ValueError),
        ({"lam": 1.0, "weight": "nope"}, ValueError),
        ({"lam": 1.0, "weight": [0.0, 1.0]}, TypeError),
        ({"lam": 1.0, "weight": "power", "p": -1}, ValueError),
    ],
)
def test_invalid_configuration(kwargs, err):
    with pytest.raises(err):
        StreamingWhittaker(**kwargs)


def test_invalid_appends():
    s = StreamingWhittaker(1.0)
    s.append(1.0, 5.0)
    with pytest.raises(ValueError, match="возрастать"):
        s.append(2.0, 5.0)
    with pytest.raises(ValueError, match="бесконечным"):
        s.append(np.inf, 6.0)
    with pytest.raises(ValueError, match="не совпадает"):
        s.extend([1.0, 2.0], times=[7.0])
    with pytest.raises(ValueError, match="нет данных"):
        StreamingWhittaker(1.0).smooth()


def test_realtime_cost_per_point(rng):
    """Окно в час при записи раз в секунду (3600 точек): append + smooth быстрее 20 мс."""
    t, y = memory_trace(rng, 4000)
    s = StreamingWhittaker(1e4, window=3600.0)
    s.extend(y[:3700], t[:3700])
    s.smooth()
    costs = []
    for v, tt in zip(y[3700:], t[3700:], strict=True):
        start = time.perf_counter()
        s.append(v, tt)
        s.smooth()
        costs.append(time.perf_counter() - start)
    assert np.median(costs) < 0.02
