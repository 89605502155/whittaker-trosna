"""Автоматический подбор параметра гладкости λ.

Четыре критерия, все — «чем меньше, тем лучше»:

* ``"cv"`` — перекрёстная проверка с исключением по одной точке (LOOCV).
  Для линейного сглаживателя ``z = H y`` ошибка без пересчёта:
  ``e_i = (y_i - z_i) / (1 - h_ii)`` (Eilers, 2003). Диагональ ``h_ii``
  вычисляется точно за ``O(n d²)`` рекурсией Такахаси.
* ``"bcv"`` — блочная перекрёстная проверка для коррелированного шума
  (h-block CV: Burman, Chow & Nolan, 1994; Chu & Marron, 1991). Соседние
  отсчёты телеметрии похожи не только из-за сигнала, но и из-за шума;
  обычная LOOCV тогда «узнаёт» исключённую точку по соседям и выбирает
  слишком малую λ (почти не сглаживает). Здесь вместе с точкой ``i``
  исключаются ``cv_block`` соседей с каждой стороны, и ошибка в ``i``
  считается по оставшимся данным. Для линейного сглаживателя это точно и
  без пересчёта: ``e_B = (I - H_BB)⁻¹ r_B``, где ``H_BB`` — блок матрицы
  сглаживания; нужные элементы ``A⁻¹`` даёт рекурсия Такахаси в полосе
  ширины ``2·cv_block`` за ``O(n d · cv_block)``.
* ``"gcv"`` — обобщённая перекрёстная проверка (Craven & Wahba, 1979):
  ``m Σ f_i r_i² / (m - tr H)²``.
* ``"forward"`` — прогнозная проверка для временных рядов: для ряда последних
  моментов ``t_i`` сглаживается только прошлое ``y_0..y_{i-1}`` и
  экстраполируется значение в ``t_i``. Критерий — RMSE прогноза. В отличие от
  LOOCV, не «подглядывает» в будущее. Подходит для классического метода; в
  варианте со свежестью прогноз привязан к закреплённой сырой последней точке,
  критерий почти не зависит от λ, и выдаётся предупреждение.

Поиск идёт по логарифмической сетке с последующим уточнением методом Брента.
Диапазон сетки выбирается по данным: от частоты среза около двух шагов
дискретизации до частоты среза порядка длительности ряда.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

from .core import SmoothingProblem, _solve, build_problem
from .weights import WeightFunction, recency_coefficients

METHODS = ("cv", "gcv", "forward", "bcv")
MAX_DEFAULT_CV_BLOCK = 10


def default_cv_block(n: int) -> int:
    """Полуширина блока для ``"bcv"`` по умолчанию: ``n // 20``, но от 1 до 10 отсчётов.

    Блок должен перекрывать корреляцию шума, но не быть сравнимым с длиной ряда: на ряде из 100 точек
    исключение 21 точки (блок 10) заставляет «угадывать» точку через большую дыру, и λ выбирается
    завышенной (ошибка в 2–3 раза больше наилучшей). Моделирование с шумом AR(1), φ = 0…0,8:
    на 100 точках лучше всего блок 2–5, на 800–1800 точках — 10.
    """
    return max(1, min(MAX_DEFAULT_CV_BLOCK, n // 20))


BCV_MAX_CENTERS = 1000
"""При подборе λ ошибка блочной проверки усредняется не более чем по стольким точкам (равномерно)."""


@dataclass(frozen=True)
class LambdaSearch:
    """Результат подбора λ: лучшее значение, критерий и вся кривая критерия по сетке."""

    lam: float
    method: str
    score: float
    lams: NDArray[np.float64]
    scores: NDArray[np.float64]


def lambda_for_cutoff(period: float, order: int = 2, dt: float | None = None) -> float:
    """λ, при которой колебания с периодом ``period`` ослабляются ровно вдвое.

    Передаточная функция сглаживателя Уиттекера на равномерной сетке с шагом
    ``h``: ``G(ω) = 1 / (1 + λ (2 sin(ω h / 2) / h)^{2d})``. Колебания короче
    ``period`` подавляются сильнее, длиннее — слабее. Без ``dt`` используется
    непрерывное приближение ``λ = (period / 2π)^{2d}``, не зависящее от шага.
    ``period`` и ``dt`` — в единицах времени ``t`` (или в шагах, если ``t`` нет).
    """
    period = float(period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"период должен быть положительным, получено {period}")
    if dt is None:
        return (period / (2 * np.pi)) ** (2 * order)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"шаг должен быть положительным, получено {dt}")
    if period <= 2 * dt:
        raise ValueError("период должен быть больше двух шагов дискретизации")
    return (dt / (2 * np.sin(np.pi * dt / period))) ** (2 * order)


def cutoff_for_lambda(lam: float, order: int = 2) -> float:
    """Период среза, соответствующий λ (непрерывное приближение): ``2π λ^{1/(2d)}``."""
    return float(2 * np.pi * lam ** (1.0 / (2 * order)))


def select_lambda(
    y: ArrayLike,
    method: str = "cv",
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weight: str | WeightFunction | ArrayLike | None = None,
    horizon: float | None = None,
    weights: ArrayLike | None = None,
    lam_range: tuple[float, float] | None = None,
    cv_block: int | None = None,
    **weight_params: Any,
) -> LambdaSearch:
    """Подобрать λ для ряда ``y``.

    ``weight=None`` — классический метод Уиттекера, иначе — вариант со
    свежестью с указанной весовой функцией (как в ``recency_whittaker``).
    ``cv_block`` — полуширина исключаемого блока для ``method="bcv"``
    (по умолчанию :func:`default_cv_block`).
    """
    if weight is None:
        problem = build_problem(y, t=t, order=order, weights=weights)
    elif isinstance(weight, str) or callable(weight):

        def rule(tt: NDArray[np.float64] | None, n: int) -> NDArray[np.float64]:
            return recency_coefficients(n, weight, t=tt, horizon=horizon, **weight_params)

        problem = build_problem(y, t=t, order=order, weights=weights, rule=rule)
    else:
        problem = build_problem(y, t=t, order=order, weights=weights, mobility=weight)
    return search_lambda(problem, method=method, lam_range=lam_range, cv_block=cv_block)


def check_method(method: str, cv_block: int | None = None) -> None:
    """Проверить способ подбора λ и полуширину блока."""
    if method not in METHODS:
        raise ValueError(f"неизвестный способ подбора λ {method!r}; доступны: {', '.join(METHODS)}")
    if cv_block is None:
        return
    if method != "bcv":
        raise ValueError(f"cv_block применим только к method='bcv', а не к {method!r}")
    if isinstance(cv_block, bool) or not isinstance(cv_block, int | np.integer) or cv_block < 1:
        raise ValueError(f"cv_block должно быть целым ≥ 1, получено {cv_block!r}")


def search_lambda(
    problem: SmoothingProblem,
    *,
    method: str = "cv",
    lam_range: tuple[float, float] | None = None,
    points_per_decade: int = 2,
    max_forward_targets: int = 40,
    cv_block: int | None = None,
) -> LambdaSearch:
    """Подбор λ для готовой задачи (см. :func:`select_lambda`)."""
    check_method(method, cv_block)
    if problem.n <= problem.order + 2:
        raise ValueError(f"для подбора λ нужно больше {problem.order + 2} точек")

    if method == "forward":
        targets = _forward_targets(problem, max_forward_targets)
        subproblems = [(problem.prefix_with_target(i), problem.y[i]) for i in targets]
        if any(sub.fixed[-2] for sub, _ in subproblems):
            warnings.warn(
                "прогнозная проверка малоинформативна, когда самая свежая точка закреплена: "
                "прогноз привязан к её сырому значению и почти не зависит от λ. "
                "Для варианта со свежестью используйте 'cv' или 'gcv'.",
                UserWarning,
                stacklevel=3,
            )

        def score(lam: float) -> float:
            return forward_score(subproblems, lam)
    elif method == "bcv":
        block = default_cv_block(problem.n) if cv_block is None else int(cv_block)
        bcv = _BlockCV(problem, block, BCV_MAX_CENTERS)

        def score(lam: float) -> float:
            return bcv.score(lam)
    else:
        crit = cv_score if method == "cv" else gcv_score

        def score(lam: float) -> float:
            return crit(problem, lam)

    lo, hi = lam_range if lam_range is not None else default_lambda_range(problem)
    if not (0 < lo < hi and np.isfinite(hi)):
        raise ValueError(f"неверный диапазон λ: {(lo, hi)}")
    log_grid = np.arange(np.log10(lo), np.log10(hi) + 1e-9, 1.0 / points_per_decade)
    scores = np.array([score(10.0**g) for g in log_grid])
    if not np.any(np.isfinite(scores)):
        raise ValueError("критерий не определён ни для одного λ")
    best = int(np.nanargmin(np.where(np.isfinite(scores), scores, np.nan)))

    best_log, best_score = log_grid[best], scores[best]
    if 0 < best < log_grid.size - 1:
        step = 1.0 / points_per_decade
        res = minimize_scalar(
            lambda g: score(10.0**g),
            bounds=(best_log - step, best_log + step),
            method="bounded",
            options={"xatol": 0.01},
        )
        if res.success and res.fun < best_score:
            best_log, best_score = float(res.x), float(res.fun)
    return LambdaSearch(float(10.0**best_log), method, float(best_score), 10.0**log_grid, scores)


def default_lambda_range(problem: SmoothingProblem) -> tuple[float, float]:
    """Диапазон λ по данным: срез от ~2 шагов дискретизации до ~4 длительностей ряда."""
    if problem.t is None:
        h, span = 1.0, float(problem.n - 1)
    else:
        h, span = float(np.median(np.diff(problem.t))), float(problem.t[-1] - problem.t[0])
    d = problem.order
    lo = lambda_for_cutoff(2.0 * h, d) * 1e-2
    hi = lambda_for_cutoff(4.0 * span, d) * 1e2
    return lo, hi


def hat_diagonal(problem: SmoothingProblem, lam: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Сглаженный ряд ``z`` и диагональ матрицы сглаживания ``h_ii`` (для закреплённых — 1)."""
    z, factor = _solve(problem, lam)
    h = np.ones(problem.n)
    if factor is not None:
        free = ~problem.fixed
        h[free] = problem.fidelity[free] * factor.inverse_diagonal()
    return z, h


def cv_score(problem: SmoothingProblem, lam: float) -> float:
    """Корень из средней квадратичной ошибки перекрёстной проверки с исключением точки."""
    z, h = hat_diagonal(problem, lam)
    use = ~problem.fixed & (problem.fidelity > 0) & (1.0 - h > 1e-10)
    if not np.any(use):
        return float("nan")
    e = (problem.y[use] - z[use]) / (1.0 - h[use])
    return float(np.sqrt(np.mean(e**2)))


def block_cv_score(
    problem: SmoothingProblem, lam: float, block: int | None = None, max_centers: int | None = None
) -> float:
    """Корень из средней квадратичной ошибки блочной перекрёстной проверки.

    Для каждой точки ``i`` с данными исключается блок ``B`` из ``i`` и ``block`` соседних свободных
    точек с каждой стороны (у краёв ряда блок обрезается), и ошибка в ``i`` берётся из
    ``y_B - z_B^{(-B)} = (I - H_BB)⁻¹ (y_B - z_B)``. При ``block = 0`` совпадает с ``"cv"``.
    Закреплённые точки (``c = 0``) не исключаются: это ограничения, а не наблюдения.
    ``max_centers`` — по скольким точкам (равномерно по ряду) усреднять ошибку; ``None`` — по всем.
    """
    if block is None:
        block = default_cv_block(problem.n)
    return _BlockCV(problem, block, max_centers).score(lam)


class _BlockCV:
    """Блочная проверка с заранее собранными индексами: при переборе λ меняются только числа."""

    def __init__(self, problem: SmoothingProblem, block: int, max_centers: int | None = None) -> None:
        self.problem = problem
        self.idx = np.flatnonzero(~problem.fixed)
        m = self.idx.size
        f = problem.fidelity[self.idx]
        self.block = block = min(int(block), (m - problem.order - 1) // 2)
        centers = np.flatnonzero(f > 0)
        if max_centers is not None and centers.size > max_centers:
            centers = centers[np.unique(np.linspace(0, centers.size - 1, max_centers).round().astype(int))]
        self.empty = block < 0 or centers.size == 0
        if self.empty:
            return
        pos = centers[:, None] + np.arange(-block, block + 1)
        self.inside = (pos >= 0) & (pos < m)
        self.pos = pos = np.clip(pos, 0, m - 1)
        a, b = pos[:, :, None], pos[:, None, :]
        # Элемент (A⁻¹)[p, q] лежит в полосе sig[min(p, q), |p - q|] (см. _Factor.inverse_band).
        self.flat = np.minimum(a, b) * (2 * block + 1) + np.abs(a - b)
        # H = A⁻¹ diag(f); позиции за краем ряда обнуляются, и в I - H_BB им отвечают единичные
        # строки и столбцы, которые на ответ не влияют.
        both = self.inside[:, :, None] & self.inside[:, None, :]
        self.scale = np.where(both, f[pos][:, None, :], 0.0)

    def score(self, lam: float) -> float:
        if self.empty:
            return float("nan")
        z, factor = _solve(self.problem, lam)
        if factor is None:
            return float("nan")
        width = 2 * self.block
        sig = np.ascontiguousarray(factor.inverse_band(width)[:, : width + 1]).reshape(-1)
        m_bb = np.eye(2 * self.block + 1) - sig.take(self.flat) * self.scale
        r = (self.problem.y - z)[self.idx]
        rhs = np.where(self.inside, r[self.pos], 0.0)
        try:
            e = np.linalg.solve(m_bb, rhs[..., None])[:, self.block, 0]
        except np.linalg.LinAlgError:
            return float("nan")
        return float(np.sqrt(np.mean(e**2)))


def gcv_score(problem: SmoothingProblem, lam: float) -> float:
    """Критерий обобщённой перекрёстной проверки (по свободным точкам с данными)."""
    z, h = hat_diagonal(problem, lam)
    use = ~problem.fixed & (problem.fidelity > 0)
    m = int(use.sum())
    trace = float(h[use].sum())
    if m == 0 or m - trace <= 1e-10:
        return float("nan")
    # Остатки взвешиваются весами наблюдений w, а не верностью w/c: 1/c — это защита
    # свежих точек, а не дисперсия шума, и с ней критерий вырождается в λ → 0.
    rss = float(np.sum(problem.w[use] * (problem.y[use] - z[use]) ** 2))
    return m * rss / (m - trace) ** 2


def forward_score(subproblems: list[tuple[SmoothingProblem, float]], lam: float) -> float:
    """RMSE прогноза следующей точки по сглаженному прошлому."""
    errors = [_solve(sub, lam)[0][-1] - target for sub, target in subproblems]
    return float(np.sqrt(np.mean(np.square(errors))))


def _forward_targets(problem: SmoothingProblem, max_targets: int) -> NDArray[np.int64]:
    """Точки для прогнозной проверки: из второй половины ряда, только с данными."""
    first = max(problem.order + 2, problem.n // 2)
    candidates = np.arange(first, problem.n)
    candidates = candidates[problem.w[candidates] > 0]
    if candidates.size == 0:
        raise ValueError("нет точек с данными для прогнозной проверки")
    if candidates.size > max_targets:
        candidates = candidates[np.unique(np.linspace(0, candidates.size - 1, max_targets).astype(int))]
    return candidates
