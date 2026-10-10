"""Как метод сглаживает: статистические проверки на зашумлённых сигналах (метод Монте-Карло)."""

import numpy as np
import pytest

from whittaker_trosna import recency_whittaker, whittaker

N = 200
REPS = 60
WEIGHTS = [("linear", {}), ("power", {"p": 3.0}), ("exponential", {"k": 8.0}), ("cosine", {})]


def thirds(a: np.ndarray) -> list[float]:
    """Среднее по трём частям ряда: старая, средняя, свежая."""
    return [float(np.mean(part)) for part in np.array_split(a, 3)]


def monte_carlo(lam, weight, params, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    truth = np.sin(np.linspace(0, 4 * np.pi, N))
    disp = np.zeros(N)
    se_rec = np.zeros(N)
    se_cls = np.zeros(N)
    se_raw = np.zeros(N)
    for _ in range(REPS):
        y = truth + noise * rng.normal(size=N)
        z = recency_whittaker(y, lam, weight=weight, **params)
        disp += np.abs(z - y)
        se_rec += (z - truth) ** 2
        se_cls += (whittaker(y, lam) - truth) ** 2
        se_raw += (y - truth) ** 2
    return disp / REPS, se_rec / REPS, se_cls / REPS, se_raw / REPS


@pytest.mark.parametrize(("weight", "params"), WEIGHTS)
@pytest.mark.parametrize("lam", [10.0, 1000.0])
def test_noise_is_reduced(weight, params, lam):
    _, se_rec, se_cls, se_raw = monte_carlo(lam, weight, params)
    rmse = lambda se: np.sqrt(thirds(se))  # noqa: E731
    # в старой части ряда — более чем двукратное подавление шума
    assert rmse(se_rec)[0] < 0.5 * rmse(se_raw)[0]
    # там же качество как у классического метода (отличие ≤ 10 %)
    assert rmse(se_rec)[0] < 1.1 * rmse(se_cls)[0]
    # по всему ряду ошибка меньше, чем у сырых данных
    assert np.sqrt(se_rec.mean()) < np.sqrt(se_raw.mean())


def displacement_profile(lam, weight, params, n=N):
    """Точное ожидаемое смещение |z - y| для белого шума: нормы строк (I - H), где z = H y."""
    h = np.column_stack([recency_whittaker(e, lam, weight=weight, **params) for e in np.eye(n)])
    return np.linalg.norm(np.eye(n) - h, axis=1)


@pytest.mark.parametrize(("weight", "params"), [w for w in WEIGHTS if w[0] != "exponential"])
def test_old_points_are_moved_more_than_fresh(weight, params):
    disp = displacement_profile(100.0, weight, params)
    old, _, fresh = thirds(disp)
    assert old > fresh
    assert disp[-1] == 0.0
    # у свежего конца смещение строго убывает к последней точке
    assert np.all(np.diff(disp[-20:]) < 0)


def test_weight_function_controls_protection_of_fresh_points():
    """Чем медленнее растёт c(t) у свежего конца, тем меньше двигаются свежие точки."""
    fresh = {
        name: thirds(displacement_profile(100.0, name, params))[2]
        for name, params in [
            ("power", {"p": 3.0}),
            ("cosine", {}),
            ("linear", {}),
            ("exponential", {"k": 8.0}),
        ]
    }
    assert fresh["power"] < fresh["cosine"] < fresh["linear"] < fresh["exponential"]


def test_neighbour_of_anchor_can_move_more_than_average():
    """Особенность метода: если c(t) быстро растёт (exponential, k=8), сосед закреплённой
    точки почти не защищён и подтягивается к «сырому» якорю сильнее, чем средняя точка."""
    disp = displacement_profile(100.0, "exponential", {"k": 8.0})
    assert disp[-2] > np.median(disp[:-1])
    # при сильной защите (power, p=3) такого нет
    disp_p = displacement_profile(100.0, "power", {"p": 3.0})
    assert disp_p[-2] < 0.01 * np.median(disp_p[:-1])


def test_roughness_decreases_with_lambda():
    rng = np.random.default_rng(5)
    y = np.sin(np.linspace(0, 6, 300)) + 0.3 * rng.normal(size=300)
    rough = [np.sum(np.diff(recency_whittaker(y, lam), 2) ** 2) for lam in (0.1, 1, 10, 100, 1e3, 1e4)]
    assert np.all(np.diff(rough) < 0)


@pytest.mark.parametrize("lam", [100.0, 1000.0])
def test_regime_shift_is_tracked_at_the_fresh_end(lam):
    """Резкая смена уровня в конце ряда: классический метод «размазывает» скачок,
    метод со свежестью сохраняет актуальное значение."""
    rng = np.random.default_rng(7)
    truth = np.r_[np.zeros(190), np.ones(10)]
    err_rec = err_cls = 0.0
    for _ in range(REPS):
        y = truth + 0.05 * rng.normal(size=200)
        err_rec += np.mean(np.abs(recency_whittaker(y, lam)[-3:] - 1.0))
        err_cls += np.mean(np.abs(whittaker(y, lam)[-3:] - 1.0))
    assert err_rec < 0.5 * err_cls


def test_price_of_fixing_newest_point_on_stationary_signal():
    """Обратная сторона: на стационарном сигнале свежий конец остаётся почти сырым,
    поэтому ошибка на последних точках выше, чем у классического метода."""
    _, se_rec, se_cls, se_raw = monte_carlo(100.0, "linear", {})
    assert se_rec[-1] == pytest.approx(se_raw[-1])
    assert se_rec[-5:].mean() > se_cls[-5:].mean()
