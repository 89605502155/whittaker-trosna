"""Сглаживание Уиттекера и его вариант со взвешиванием по свежести точек.

Классический метод Уиттекера (Whittaker, 1923; Eilers, 2003) ищет ряд ``z``,
минимизирующий

    J(z) = Σ w_i (y_i - z_i)²  +  λ Σ (Δ^d z)_i²,

где первое слагаемое — верность данным, второе — шероховатость, ``Δ^d`` —
конечная разность порядка ``d`` (для неравномерного шага по времени —
разделённая разность, Eilers, 2003). Оптимум даётся линейной системой
``(W + λ DᵀD) z = W y`` с ленточной матрицей.

Вариант со свежестью вводит для каждой точки коэффициент подвижности
``c_i ∈ [0, 1]`` и минимизирует

    J(z) = Σ_{c_i > 0} w_i (y_i - z_i)² / c_i  +  λ Σ (Δ^d z)_i²
    при ограничениях  z_i = y_i  для всех c_i = 0, w_i > 0.

* ``c_i = 1`` — точка сглаживается с классической силой;
* ``c_i → 0`` — штраф за сдвиг растёт как ``1/c_i``, и в пределе точка
  закрепляется (``z_i = y_i``);
* ``w_i = 0`` — пропуск: значение восстанавливается по соседям.

При ``c ≡ 1`` метод совпадает с классическим. Ограничения устраняются
подстановкой, и для свободных точек ``F`` решается симметричная положительно
определённая ленточная система

    (diag(f_F) + λ P_FF) z_F = f_F y_F - λ P_FK y_K,   P = DᵀD,  f = w / c,

где ``K`` — закреплённые точки. Главная подматрица ленточной матрицы снова
ленточная с той же шириной ленты, поэтому сложность — ``O(n d²)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.special import comb

from .weights import WeightFunction, check_times, recency_coefficients

LambdaSpec = float | str


def difference_matrix(n: int, order: int = 2, t: ArrayLike | None = None) -> sp.csr_matrix:
    """Матрица разностей ``D`` размера ``(n - order) × n``.

    Без ``t`` — обычные конечные разности. С ``t`` — разделённые разности
    по Эйлерсу (``ddmat``): ``D_k = diag(k / (t_{i+k} - t_i)) · Δ D_{k-1}``;
    при единичном шаге они совпадают с обычными.
    """
    _check_order(order)
    if n <= order:
        return sp.csr_matrix((0, n))
    if t is None:
        k = np.arange(order + 1)
        coeffs = (-1.0) ** (order - k) * comb(order, k)
        return sp.diags(coeffs, k, shape=(n - order, n), format="csr")
    t = check_times(t, n)
    d = sp.identity(n, format="csr")
    for k in range(1, order + 1):
        m = n - k
        diff = sp.diags([-np.ones(m), np.ones(m)], [0, 1], shape=(m, m + 1), format="csr")
        d = sp.diags(k / (t[k:] - t[:-k])) @ diff @ d
    return d.tocsr()


def penalty_matrix(n: int, order: int = 2, t: ArrayLike | None = None) -> sp.csr_matrix:
    """Матрица штрафа ``P = DᵀD`` (ленточная, ширина ленты ``order``)."""
    d = difference_matrix(n, order, t)
    return (d.T @ d).tocsr()


def difference_bands(n: int, order: int = 2, t: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    """Ненулевые элементы ``D`` построчно: ``C[i, j] = D[i, i + j]``, форма ``(n - order, order + 1)``."""
    if n <= order:
        return np.zeros((0, order + 1))
    c = np.ones((n, 1))
    for k in range(1, order + 1):
        m = n - k
        nxt = np.zeros((m, k + 1))
        nxt[:, :k] -= c[:m]
        nxt[:, 1:] += c[1 : m + 1]
        if t is not None:
            nxt *= (k / (t[k:] - t[:-k]))[:, None]
        c = nxt
    return c


def penalty_bands(c: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Наддиагонали ``P = DᵀD``: ``Pb[m, i] = P[i, i + m]``, форма ``(order + 1, n)``."""
    rows, width = c.shape
    pb = np.zeros((width, n))
    for j in range(width):
        for m in range(width - j):
            pb[m, j : j + rows] += c[:, j] * c[:, j + m]
    return pb


def band_matvec(pb: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Произведение симметричной ленточной матрицы (в форме ``penalty_bands``) на вектор."""
    out = pb[0] * x
    for m in range(1, pb.shape[0]):
        if m >= x.size:
            break
        out[:-m] += pb[m, :-m] * x[m:]
        out[m:] += pb[m, :-m] * x[:-m]
    return out


def _sub_bands(pb: NDArray[np.float64], idx: NDArray[np.int64]) -> NDArray[np.float64]:
    """Наддиагонали главной подматрицы ``P[idx][:, idx]`` (тоже ленточная, ширина не больше)."""
    width = pb.shape[0]
    m_sub = idx.size
    sub = np.zeros((width, m_sub))
    sub[0] = pb[0, idx]
    for m in range(1, min(width, m_sub)):
        a = np.arange(m_sub - m)
        dist = idx[a + m] - idx[a]
        ok = dist < width
        sub[m, a[ok]] = pb[dist[ok], idx[a[ok]]]
    return sub


MobilityRule = Callable[[NDArray[np.float64] | None, int], NDArray[np.float64]]


class SmoothingProblem:
    """Задача сглаживания в общем виде.

    Хранит исходные данные ``y``, веса верности ``w``, коэффициенты подвижности
    ``c``, моменты времени ``t`` и (необязательно) правило ``rule(t, n) -> c``,
    по которому коэффициенты пересчитываются для части ряда — это нужно для
    прогнозной проверки, где «самой свежей» становится каждая точка по очереди.
    """

    def __init__(
        self,
        y: NDArray[np.float64],
        w: NDArray[np.float64],
        c: NDArray[np.float64],
        *,
        t: NDArray[np.float64] | None,
        order: int,
        rule: MobilityRule | None = None,
    ) -> None:
        self.y, self.w, self.c, self.t, self.order, self.rule = y, w, c, t, order, rule
        self.fixed = (c == 0) & (w > 0)
        self.fidelity = np.zeros(y.size)
        movable = c > 0
        self.fidelity[movable] = w[movable] / c[movable]
        self.pband = penalty_bands(difference_bands(y.size, order, t), y.size)

    @property
    def n(self) -> int:
        return self.y.size

    def prefix_with_target(self, i: int) -> SmoothingProblem:
        """Задача «в момент ``t[i-1]``»: точки ``0..i-1`` и точка ``i`` без данных (вес 0)."""
        t_pre = None if self.t is None else self.t[:i]
        c_pre = self.rule(t_pre, i) if self.rule is not None else self.c[:i]
        c = np.r_[c_pre, 1.0]
        w = np.r_[self.w[:i], 0.0]
        t = None if self.t is None else self.t[: i + 1]
        return SmoothingProblem(self.y[: i + 1], w, c, t=t, order=self.order)


@dataclass(frozen=True)
class SmoothResult:
    """Результат сглаживания: ряд и использованное значение λ."""

    z: NDArray[np.float64]
    lam: float
    mobility: NDArray[np.float64] | None = None
    search: Any = None  # LambdaSearch, если λ подбиралась автоматически


def build_problem(
    y: ArrayLike,
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weights: ArrayLike | None = None,
    mobility: ArrayLike | None = None,
    rule: MobilityRule | None = None,
) -> SmoothingProblem:
    """Собрать задачу из данных, моментов времени, весов и коэффициентов подвижности."""
    y = _check_series(y)
    _check_order(order)
    n = y.size
    w = _check_weights(weights, n)
    t_arr = None if t is None else check_times(t, n)
    if mobility is not None:
        c = recency_coefficients(n, mobility)
    elif rule is not None:
        c = rule(t_arr, n)
    else:
        c = np.ones(n)
    return SmoothingProblem(y, w, c, t=t_arr, order=order, rule=rule)


def solve_problem(problem: SmoothingProblem, lam: float) -> NDArray[np.float64]:
    """Решить задачу при заданной λ."""
    z, _ = _solve(problem, _check_lambda(lam))
    return z


def whittaker(
    y: ArrayLike,
    lam: LambdaSpec,
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weights: ArrayLike | None = None,
    strength: float = 1.0,
    cv_block: int | None = None,
) -> NDArray[np.float64]:
    """Классическое сглаживание Уиттекера.

    ``lam`` — число или способ автоматического подбора (``"cv"``, ``"gcv"``,
    ``"forward"``, ``"bcv"``, см. :func:`whittaker_trosna.select_lambda`).
    ``t`` — моменты времени для неравномерного шага. ``weights[i] = 0``
    означает пропуск. ``strength`` умножает итоговую λ. ``cv_block`` —
    полуширина исключаемого блока для ``"bcv"``.
    """
    problem = build_problem(y, t=t, order=order, weights=weights)
    return _fit(problem, lam, strength, cv_block).z


def recency_whittaker(
    y: ArrayLike,
    lam: LambdaSpec,
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weight: str | WeightFunction | ArrayLike = "linear",
    horizon: float | None = None,
    weights: ArrayLike | None = None,
    strength: float = 1.0,
    cv_block: int | None = None,
    **weight_params: Any,
) -> NDArray[np.float64]:
    """Сглаживание Уиттекера, взвешенное по свежести точек ряда.

    Параметры
    ---------
    y
        Временной ряд, упорядоченный от старых значений к новым
        (``y[-1]`` — самое свежее).
    lam
        Параметр гладкости ``λ ≥ 0`` или способ его автоматического подбора:
        ``"cv"`` (перекрёстная проверка с исключением по одной точке),
        ``"gcv"`` (обобщённая перекрёстная проверка), ``"forward"`` (ошибка
        прогноза следующей точки только по прошлому), ``"bcv"`` (блочная
        перекрёстная проверка, устойчивая к коррелированному шуму).
    t
        Моменты времени точек (строго возрастают). Задают неравномерный шаг:
        разности в штрафе делятся на реальные промежутки времени, а возраст
        точки измеряется во времени, а не в номерах.
    order
        Порядок разности в штрафе (обычно 2).
    weight
        Весовая функция свежести: имя (``"linear"``, ``"power"``,
        ``"exponential"``, ``"cosine"``, ``"logistic"``), вызываемый объект
        ``f(age) -> c`` от нормированного возраста ``age ∈ [0, 1]`` или готовый
        массив коэффициентов подвижности ``c ∈ [0, 1]``.
    horizon
        Возраст (в единицах ``t``), начиная с которого точка сглаживается с
        полной силой. По умолчанию — длительность всего ряда.
    weights
        Веса верности ``w_i ≥ 0``; ``0`` — пропуск.
    strength
        Сила сглаживания: множитель к λ. ``1`` — λ как задана (или как
        подобрана автоматически), ``10`` — в 10 раз сильнее, ``0.1`` — мягче.
    cv_block
        Для ``lam="bcv"``: сколько соседей с каждой стороны исключать вместе
        с точкой. По умолчанию ``n // 20``, от 1 до 10. Увеличивайте, если
        шум коррелирован дольше; блок не должен быть сравним с длиной ряда.
    **weight_params
        Параметры весовой функции, например ``p=3`` для ``"power"``.

    Возвращает
    ----------
    Сглаженный ряд той же длины. Точки с ``c = 0`` (для встроенных функций —
    последняя точка) не изменяются.
    """
    return fit_recency_whittaker(
        y,
        lam,
        t=t,
        order=order,
        weight=weight,
        horizon=horizon,
        weights=weights,
        strength=strength,
        cv_block=cv_block,
        **weight_params,
    ).z


def fit_recency_whittaker(
    y: ArrayLike,
    lam: LambdaSpec,
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weight: str | WeightFunction | ArrayLike = "linear",
    horizon: float | None = None,
    weights: ArrayLike | None = None,
    strength: float = 1.0,
    cv_block: int | None = None,
    **weight_params: Any,
) -> SmoothResult:
    """То же, что :func:`recency_whittaker`, но возвращает ещё λ и подбор λ."""
    if isinstance(weight, str) or callable(weight):

        def rule(tt: NDArray[np.float64] | None, n: int) -> NDArray[np.float64]:
            return recency_coefficients(n, weight, t=tt, horizon=horizon, **weight_params)

        problem = build_problem(y, t=t, order=order, weights=weights, rule=rule)
    else:
        if weight_params:
            raise TypeError("параметры весовой функции не применимы к готовому массиву")
        problem = build_problem(y, t=t, order=order, weights=weights, mobility=weight)
    result = _fit(problem, lam, strength, cv_block)
    return SmoothResult(result.z, result.lam, problem.c, result.search)


def smooth_with_mobility(
    y: ArrayLike,
    lam: LambdaSpec,
    mobility: ArrayLike,
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weights: ArrayLike | None = None,
    strength: float = 1.0,
    cv_block: int | None = None,
) -> NDArray[np.float64]:
    """Ядро метода: сглаживание с заданными коэффициентами подвижности ``c``."""
    problem = build_problem(y, t=t, order=order, weights=weights, mobility=mobility)
    return _fit(problem, lam, strength, cv_block).z


def objective(
    y: ArrayLike,
    z: ArrayLike,
    lam: float,
    mobility: ArrayLike,
    *,
    t: ArrayLike | None = None,
    order: int = 2,
    weights: ArrayLike | None = None,
) -> float:
    """Значение функционала ``J(z)``; ``inf``, если нарушено ``z_i = y_i`` для закреплённых точек."""
    problem = build_problem(y, t=t, order=order, weights=weights, mobility=mobility)
    z = np.asarray(z, dtype=np.float64)
    if not np.allclose(z[problem.fixed], problem.y[problem.fixed], rtol=0.0, atol=1e-12):
        return float("inf")
    fidelity = np.sum(problem.fidelity * (problem.y - z) ** 2)
    roughness = float(z @ band_matvec(problem.pband, z))
    return float(fidelity + lam * roughness)


def _fit(
    problem: SmoothingProblem, lam: LambdaSpec, strength: float, cv_block: int | None = None
) -> SmoothResult:
    strength = float(strength)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(f"сила сглаживания должна быть положительной, получено {strength}")
    search = None
    if isinstance(lam, str):
        from .lambda_selection import search_lambda  # noqa: PLC0415  (взаимный импорт)

        search = search_lambda(problem, method=lam, cv_block=cv_block)
        base = search.lam
    else:
        if cv_block is not None:
            raise ValueError("cv_block применим только при подборе λ способом 'bcv'")
        base = _check_lambda(lam)
    lam_final = base * strength
    z, _ = _solve(problem, lam_final)
    return SmoothResult(z, lam_final, None, search)


def _solve(problem: SmoothingProblem, lam: float) -> tuple[NDArray[np.float64], _Factor | None]:
    """Решение задачи; вместе с ним возвращается разложение Холецкого (для подбора λ)."""
    y, f, fixed, pb = problem.y, problem.fidelity, problem.fixed, problem.pband
    free = ~fixed
    idx = np.flatnonzero(free)
    if idx.size == 0:
        return y.copy(), None
    if lam == 0.0 or problem.n <= problem.order:
        if np.any(f[idx] == 0):
            raise ValueError("без штрафа точки с нулевым весом не восстановить")
        return y.copy(), None

    rhs = f[idx] * y[idx]
    if np.any(fixed):
        rhs = rhs - lam * band_matvec(pb, np.where(fixed, y, 0.0))[idx]
    sub = _sub_bands(pb, idx)
    b = min(problem.order, idx.size - 1)
    ab = np.zeros((b + 1, idx.size))
    for m in range(b + 1):
        ab[b - m, m:] = lam * sub[m, : idx.size - m]
    ab[b] += f[idx]
    factor = _Factor.from_band(ab)
    z = y.copy()
    z[idx] = factor.solve(rhs)
    return z, factor


@dataclass(frozen=True)
class _Factor:
    """Ленточное разложение Холецкого ``A = UᵀU`` (``U`` — верхняя ленточная)."""

    u_band: NDArray[np.float64]  # форма (b + 1, m), u_band[b + i - j, j] = U[i, j]
    bandwidth: int

    @classmethod
    def from_matrix(cls, a: sp.spmatrix, bandwidth: int) -> _Factor:
        a = sp.csr_matrix(a)
        m = a.shape[0]
        b = min(bandwidth, m - 1)
        ab = np.zeros((b + 1, m))
        for k in range(b + 1):
            ab[b - k, k:] = a.diagonal(k)
        return cls.from_band(ab)

    @classmethod
    def from_band(cls, ab: NDArray[np.float64]) -> _Factor:
        """Разложение по верхней ленточной форме LAPACK: ``ab[b + i - j, j] = A[i, j]``."""
        b = ab.shape[0] - 1
        try:
            u = scipy.linalg.cholesky_banded(ab, lower=False, check_finite=False)
        except np.linalg.LinAlgError:
            raise ValueError(
                "система вырождена: слишком мало точек с положительным весом для такого порядка разности"
            ) from None
        return cls(u, b)

    def solve(self, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        return scipy.linalg.cho_solve_banded((self.u_band, False), rhs, check_finite=False)

    def inverse_diagonal(self) -> NDArray[np.float64]:
        """Диагональ ``A⁻¹`` за ``O(m b²)`` (рекурсия Такахаси для ленточной матрицы).

        Из ``U Σ = U⁻ᵀ`` (нижнетреугольная с диагональю ``1/U_ii``) для ``j ≥ i``:
        ``Σ_ij = (δ_ij / U_ii - Σ_{k>i} U_ik Σ_kj) / U_ii``.
        """
        b = self.bandwidth
        ub = self.u_band
        m = ub.shape[1]
        diag_u = ub[b].tolist()
        # off[k][i] = U[i, i + k], k = 1..b
        off = [None] + [np.r_[ub[b - k, k:], np.zeros(k)].tolist() for k in range(1, b + 1)]
        # sig[k][i] = Σ[i, i + k]
        sig = [[0.0] * m for _ in range(b + 1)]
        for i in range(m - 1, -1, -1):
            uii = diag_u[i]
            kmax = min(b, m - 1 - i)
            for j in range(kmax, -1, -1):  # Σ[i, i + j]
                acc = 0.0
                for k in range(1, kmax + 1):  # U[i, i + k] * Σ[i + k, i + j]
                    lo, dist = (i + k, j - k) if j >= k else (i + j, k - j)
                    acc += off[k][i] * sig[dist][lo]
                sig[j][i] = ((1.0 / uii if j == 0 else 0.0) - acc) / uii
        return np.asarray(sig[0])

    def inverse_band(self, width: int) -> NDArray[np.float64]:
        """Элементы ``A⁻¹`` в полосе ширины ``width``: ``S[i, j] = (A⁻¹)[i, i + j]``, ``j = 0..width``.

        Та же рекурсия Такахаси, что и в :meth:`inverse_diagonal`, но для более широкой полосы:
        элементы ``Σ_ij`` при ``|i - j| ≤ width`` выражаются только через элементы той же полосы.
        Стоимость ``O(m · b · width)``; за пределами матрицы (``i + j ≥ m``) — нули.
        """
        b = self.bandwidth
        ub = self.u_band
        m = ub.shape[1]
        width = max(int(width), b)
        diag_u = ub[b]
        # u_off[i, k - 1] = U[i, i + k], k = 1..b
        u_off = np.zeros((m, b))
        for k in range(1, b + 1):
            u_off[: m - k, k - 1] = ub[b - k, k:]
        w1 = width + 1
        sig = np.zeros((m + width + 1, w1))  # запас строк — нули за пределами матрицы
        flat = sig.reshape(-1)
        # Σ[i + k, i + j] (j = 1..width, k = 1..b) лежит в sig[i + min(j, k), |j - k|],
        # то есть в flat[i·w1 + off[k - 1, j - 1]] со сдвигом off, не зависящим от i.
        jj = np.arange(1, width + 1)
        kk = np.arange(1, b + 1)[:, None]
        off = np.minimum(jj, kk) * w1 + np.abs(jj - kk)
        for i in range(m - 1, -1, -1):
            uii = diag_u[i]
            u = u_off[i]
            row = sig[i]
            row[1:] = (u @ flat.take(off + i * w1)) / -uii
            row[0] = (1.0 / uii - u @ row[1 : b + 1]) / uii
        out = sig[:m]
        out[np.arange(m)[:, None] + np.arange(width + 1) >= m] = 0.0
        return out


def _check_series(y: ArrayLike) -> NDArray[np.float64]:
    y = np.array(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"ожидался одномерный ряд, получена форма {y.shape}")
    if y.size == 0:
        raise ValueError("ряд пуст")
    if not np.all(np.isfinite(y)):
        raise ValueError("ряд содержит NaN или бесконечность")
    return y


def _check_weights(weights: ArrayLike | None, n: int) -> NDArray[np.float64]:
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"ожидалось {n} весов, получена форма {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("веса должны быть конечными и неотрицательными")
    return w


def _check_lambda(lam: float) -> float:
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"λ должна быть конечной и неотрицательной, получено {lam}")
    return lam


def _check_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, int | np.integer) or order < 1:
        raise ValueError(f"порядок разности должен быть целым ≥ 1, получено {order!r}")
