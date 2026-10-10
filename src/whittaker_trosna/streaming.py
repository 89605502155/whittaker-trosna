"""Потоковое сглаживание: точки дописываются по одной, сглаживание подстраивается.

Типичный сценарий — телеметрия гипервизора: раз в одну или несколько секунд
приходит значение потребления памяти, а перед подачей в регрессионную модель
нужен сглаженный ряд за последнее окно.

При взвешивании по свежести с каждой новой точкой меняется возраст всех
остальных, а значит, и все коэффициенты подвижности, поэтому точное
обновление «за O(1)» невозможно: окно пересчитывается целиком. Решатель
ленточный, поэтому пересчёт стоит ``O(W d²)`` для окна из ``W`` точек
(доли миллисекунды для окна в несколько тысяч точек). Пересчёт ленивый:
он выполняется при запросе результата, только если появились новые данные.
Автоматический подбор λ дороже, поэтому он повторяется раз в
``retune_every`` новых точек, а между подборами λ переиспользуется.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import LambdaSpec, SmoothingProblem, _check_lambda, _solve, build_problem
from .lambda_selection import LambdaSearch, check_method, search_lambda
from .weights import WeightFunction, recency_coefficients


@dataclass(frozen=True)
class StreamSnapshot:
    """Сглаженное состояние окна: моменты, исходные значения (NaN — пропуск), сглаженный ряд, λ."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    lam: float | None
    search: LambdaSearch | None = None


class StreamingWhittaker:
    """Потоковый сглаживатель Уиттекера со скользящим окном и весами по свежести.

    Параметры
    ---------
    lam
        λ (число) или способ автоподбора (``"cv"``, ``"gcv"``, ``"forward"``, ``"bcv"``).
        Для телеметрии с коррелированным шумом — ``"bcv"``.
    order
        Порядок разности в штрафе.
    weight
        Весовая функция свежести (как в ``recency_whittaker``) или ``None`` —
        классическое сглаживание без учёта свежести.
    horizon
        Возраст (в единицах времени), начиная с которого точка сглаживается с
        полной силой. По умолчанию — длительность окна.
    strength
        Множитель к λ (сила сглаживания).
    window
        Длительность окна в единицах времени: хранятся точки не старше
        ``t_последней - window``.
    max_points
        Максимальное число точек в окне.
    retune_every
        Через сколько новых точек заново подбирать λ (для автоподбора).
    cv_block
        Полуширина исключаемого блока для ``"bcv"`` (в отсчётах). По умолчанию ``n // 20``
        для текущего окна, от 1 до 10.
    **weight_params
        Параметры весовой функции.
    """

    def __init__(
        self,
        lam: LambdaSpec,
        *,
        order: int = 2,
        weight: str | WeightFunction | None = "linear",
        horizon: float | None = None,
        strength: float = 1.0,
        window: float | None = None,
        max_points: int | None = None,
        retune_every: int = 60,
        cv_block: int | None = None,
        **weight_params: Any,
    ) -> None:
        if isinstance(lam, str):
            check_method(lam, cv_block)
        else:
            _check_lambda(lam)
            if cv_block is not None:
                raise ValueError("cv_block применим только при подборе λ способом 'bcv'")
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError(f"сила сглаживания должна быть положительной, получено {strength}")
        if window is not None and not (np.isfinite(window) and window > 0):
            raise ValueError(f"длительность окна должна быть положительной, получено {window}")
        if max_points is not None and max_points < 1:
            raise ValueError(f"max_points должно быть ≥ 1, получено {max_points}")
        if retune_every < 1:
            raise ValueError(f"retune_every должно быть ≥ 1, получено {retune_every}")
        if weight is not None and not (isinstance(weight, str) or callable(weight)):
            raise TypeError("weight должен быть именем весовой функции, функцией или None")
        if weight is not None:
            recency_coefficients(3, weight, horizon=horizon, **weight_params)  # ранняя проверка параметров

        self.lam_spec = lam
        self.order = order
        self.weight = weight
        self.horizon = horizon
        self.strength = float(strength)
        self.window = window
        self.max_points = max_points
        self.retune_every = retune_every
        self.cv_block = cv_block
        self.weight_params = weight_params

        self._t: deque[float] = deque()
        self._y: deque[float] = deque()
        self._base_lam: float | None = None if isinstance(lam, str) else float(lam)
        self._search: LambdaSearch | None = None
        self._since_tune = 0
        self._snapshot: StreamSnapshot | None = None

    # --- добавление данных -------------------------------------------------

    def append(self, value: float, t: float | None = None) -> None:
        """Добавить точку. ``t`` по умолчанию — предыдущий момент + 1; ``NaN`` — пропуск."""
        value = float(value)
        if np.isinf(value):
            raise ValueError("значение не может быть бесконечным")
        if t is None:
            t = self._t[-1] + 1.0 if self._t else 0.0
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("момент времени должен быть конечным")
        if self._t and t <= self._t[-1]:
            raise ValueError(f"моменты времени должны строго возрастать: {t} после {self._t[-1]}")
        self._t.append(t)
        self._y.append(value)
        self._trim()
        self._since_tune += 1
        self._snapshot = None

    def extend(self, values: ArrayLike, times: ArrayLike | None = None) -> None:
        """Добавить несколько точек подряд."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if times is None:
            for v in values:
                self.append(v)
            return
        times = np.asarray(times, dtype=np.float64).ravel()
        if times.shape != values.shape:
            raise ValueError("число моментов времени не совпадает с числом значений")
        for v, tt in zip(values, times, strict=True):
            self.append(v, tt)

    def _trim(self) -> None:
        if self.window is not None:
            limit = self._t[-1] - self.window
            while self._t[0] < limit:
                self._t.popleft()
                self._y.popleft()
        if self.max_points is not None:
            while len(self._t) > self.max_points:
                self._t.popleft()
                self._y.popleft()

    # --- результат ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._t)

    @property
    def lam(self) -> float | None:
        """Текущая итоговая λ (с учётом силы сглаживания) или ``None``, пока она не определена."""
        return None if self._base_lam is None else self._base_lam * self.strength

    def smooth(self) -> StreamSnapshot:
        """Сглаженное состояние текущего окна (пересчитывается только при новых данных)."""
        if self._snapshot is not None:
            return self._snapshot
        if not self._t:
            raise ValueError("нет данных")
        t = np.fromiter(self._t, dtype=np.float64)
        y_raw = np.fromiter(self._y, dtype=np.float64)
        missing = np.isnan(y_raw)
        y = np.where(missing, 0.0, y_raw)
        w = (~missing).astype(np.float64)

        if t.size <= self.order + 2 or w.sum() <= self.order:
            # Слишком мало данных для сглаживания — возвращаем как есть.
            self._snapshot = StreamSnapshot(t, y_raw, y_raw.copy(), self.lam)
            return self._snapshot

        problem = self._problem(y, t, w)
        if isinstance(self.lam_spec, str) and (
            self._base_lam is None or self._since_tune >= self.retune_every
        ):
            self._search = search_lambda(problem, method=self.lam_spec, cv_block=self.cv_block)
            self._base_lam = self._search.lam
            self._since_tune = 0
        z, _ = _solve(problem, self.lam)
        self._snapshot = StreamSnapshot(t, y_raw, z, self.lam, self._search)
        return self._snapshot

    @property
    def latest(self) -> float:
        """Сглаженное значение в самой свежей точке."""
        return float(self.smooth().z[-1])

    def _problem(self, y: NDArray, t: NDArray, w: NDArray) -> SmoothingProblem:
        if self.weight is None:
            return build_problem(y, t=t, order=self.order, weights=w)
        weight, horizon, params = self.weight, self.horizon, self.weight_params

        def rule(tt: NDArray[np.float64] | None, n: int) -> NDArray[np.float64]:
            return recency_coefficients(n, weight, t=tt, horizon=horizon, **params)

        return build_problem(y, t=t, order=self.order, weights=w, rule=rule)
