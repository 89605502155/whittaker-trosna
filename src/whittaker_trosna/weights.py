"""Весовые функции «свежести» точки временного ряда.

Коэффициент подвижности ``c_i`` задаёт, насколько сглаживание может сдвинуть
точку ``i``:

* ``c = 0`` — точку сдвигать нельзя (самая свежая точка ряда);
* ``c = 1`` — точка сглаживается как в классическом методе Уиттекера
  (самая старая точка ряда).

Каждая весовая функция ``f`` отображает нормированное расстояние от самой
свежей точки ``t ∈ [0, 1]`` в коэффициент ``c ∈ [0, 1]``, причём
``f(0) = 0``, ``f(1) = 1`` и ``f`` монотонно не убывает.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

WeightFunction = Callable[[NDArray[np.float64]], ArrayLike]


def normalized_age(n: int) -> NDArray[np.float64]:
    """Нормированное расстояние каждой точки от самой свежей.

    Точка с индексом ``n - 1`` (последняя, самая свежая) получает 0,
    точка с индексом 0 (самая старая) получает 1.
    """
    if n < 1:
        raise ValueError(f"длина ряда должна быть положительной, получено {n}")
    if n == 1:
        return np.zeros(1)
    return (n - 1 - np.arange(n, dtype=np.float64)) / (n - 1)


def linear(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """``c = t``."""
    return np.asarray(t, dtype=np.float64).copy()


def power(t: NDArray[np.float64], p: float = 2.0) -> NDArray[np.float64]:
    """``c = t**p``. При ``p > 1`` защищено больше свежих точек, при ``p < 1`` меньше."""
    if not np.isfinite(p) or p <= 0:
        raise ValueError(f"показатель p должен быть положительным, получено {p}")
    return np.asarray(t, dtype=np.float64) ** p


def exponential(t: NDArray[np.float64], k: float = 5.0) -> NDArray[np.float64]:
    """``c = (1 - exp(-k t)) / (1 - exp(-k))``.

    При ``k > 0`` коэффициент быстро растёт, т.е. защищены только самые свежие
    точки; при ``k < 0`` растёт медленно; при ``k → 0`` функция становится
    линейной.
    """
    if not np.isfinite(k):
        raise ValueError(f"параметр k должен быть конечным, получено {k}")
    t = np.asarray(t, dtype=np.float64)
    if abs(k) < 1e-12:
        return t.copy()
    return np.expm1(-k * t) / np.expm1(-k)


def cosine(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """``c = (1 - cos(π t)) / 2``: плавный переход с нулевой производной на концах."""
    t = np.asarray(t, dtype=np.float64)
    return (1.0 - np.cos(np.pi * t)) / 2.0


def logistic(t: NDArray[np.float64], center: float = 0.5, steepness: float = 10.0) -> NDArray[np.float64]:
    """Логистическая кривая с центром ``center``, перенормированная на ``[0, 1]``."""
    if not 0.0 <= center <= 1.0:
        raise ValueError(f"center должен лежать в [0, 1], получено {center}")
    if not np.isfinite(steepness) or steepness <= 0:
        raise ValueError(f"steepness должна быть положительной, получено {steepness}")
    t = np.asarray(t, dtype=np.float64)

    def s(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-steepness * (np.asarray(x) - center)))

    s0, s1 = s(0.0), s(1.0)
    return (s(t) - s0) / (s1 - s0)


WEIGHT_FUNCTIONS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "linear": linear,
    "power": power,
    "exponential": exponential,
    "cosine": cosine,
    "logistic": logistic,
}


def normalized_age_from_times(
    t: ArrayLike | None, n: int, horizon: float | None = None
) -> NDArray[np.float64]:
    """Нормированный возраст точек ``∈ [0, 1]`` с учётом реальных моментов времени.

    Возраст — время, прошедшее от момента точки до самой свежей точки ``t[-1]``.
    Без ``horizon`` он делится на длительность всего ряда (самая старая точка
    получает 1). С ``horizon`` — делится на ``horizon`` и обрезается сверху
    единицей: точки старше горизонта сглаживаются с полной силой. Если ``t``
    не задано, моментами считаются номера точек ``0, 1, …, n-1``.
    """
    if n < 1:
        raise ValueError(f"длина ряда должна быть положительной, получено {n}")
    tt = np.arange(n, dtype=np.float64) if t is None else check_times(t, n)
    age = tt[-1] - tt
    if horizon is None:
        scale = age[0]
    else:
        scale = float(horizon)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"горизонт должен быть положительным, получено {horizon}")
    if scale == 0:
        return np.zeros(n)
    return np.clip(age / scale, 0.0, 1.0)


def check_times(t: ArrayLike, n: int) -> NDArray[np.float64]:
    """Проверка моментов времени: ``n`` конечных строго возрастающих чисел."""
    t = np.array(t, dtype=np.float64)
    if t.shape != (n,):
        raise ValueError(f"ожидалось {n} моментов времени, получена форма {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("моменты времени содержат NaN или бесконечность")
    if np.any(np.diff(t) <= 0):
        raise ValueError("моменты времени должны строго возрастать")
    return t


def recency_coefficients(
    n: int,
    weight: str | WeightFunction | ArrayLike = "linear",
    *,
    t: ArrayLike | None = None,
    horizon: float | None = None,
    **params: Any,
) -> NDArray[np.float64]:
    """Коэффициенты подвижности ``c`` для ряда длины ``n``.

    ``weight`` может быть:

    * именем встроенной функции (см. ``WEIGHT_FUNCTIONS``), ``params``
      передаются ей как параметры;
    * вызываемым объектом ``f(age) -> c``, где ``age`` — результат
      :func:`normalized_age_from_times`;
    * готовым массивом коэффициентов длины ``n`` (тогда ``t`` и ``horizon``
      не используются).

    ``t`` — моменты времени точек (для неравномерного шага), ``horizon`` —
    возраст, начиная с которого точка сглаживается с полной силой (в единицах
    ``t``). Результат проверяется: все значения конечны и лежат в ``[0, 1]``.
    """
    if isinstance(weight, str):
        try:
            func = WEIGHT_FUNCTIONS[weight]
        except KeyError:
            known = ", ".join(sorted(WEIGHT_FUNCTIONS))
            raise ValueError(f"неизвестная весовая функция {weight!r}; доступны: {known}") from None
        c = func(normalized_age_from_times(t, n, horizon), **params)
    elif callable(weight):
        c = weight(normalized_age_from_times(t, n, horizon), **params)
    else:
        if params:
            raise TypeError("параметры весовой функции не применимы к готовому массиву")
        c = weight

    c = np.asarray(c, dtype=np.float64)
    if c.shape != (n,):
        raise ValueError(f"ожидалось {n} коэффициентов, получена форма {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("коэффициенты содержат NaN или бесконечность")
    # Устраняем ошибки округления на границах (например, 1.0000000000000002).
    tol = 1e-12
    if np.any(c < -tol) or np.any(c > 1 + tol):
        raise ValueError("коэффициенты должны лежать в отрезке [0, 1]")
    return np.clip(c, 0.0, 1.0)
