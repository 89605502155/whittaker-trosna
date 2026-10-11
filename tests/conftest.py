"""Общие эталонные реализации для тестов (плотные, без оптимизаций)."""

import numpy as np
import pytest


def dense_difference(n: int, order: int) -> np.ndarray:
    return np.diff(np.eye(n), n=order, axis=0)


def reference_classic(y: np.ndarray, lam: float, order: int, w: np.ndarray | None = None):
    """Классический Уиттекер прямым плотным решением (W + λDᵀD) z = W y."""
    n = y.size
    w = np.ones(n) if w is None else w
    d = dense_difference(n, order)
    return np.linalg.solve(np.diag(w) + lam * d.T @ d, w * y)


def reference_recency(y: np.ndarray, lam: float, c: np.ndarray, order: int) -> np.ndarray:
    """Условная минимизация через систему ККТ с множителями Лагранжа.

    min Σ_{c>0} (y-z)²/c + λ‖Dz‖²  при  z_K = y_K.
    Независима от схемы исключения переменных в библиотеке.
    """
    n = y.size
    d = dense_difference(n, order)
    free = c > 0
    w = np.zeros(n)
    w[free] = 1.0 / c[free]
    h = 2.0 * (np.diag(w) + lam * d.T @ d)
    fixed_idx = np.flatnonzero(~free)
    e = np.zeros((fixed_idx.size, n))
    e[np.arange(fixed_idx.size), fixed_idx] = 1.0
    kkt = np.block([[h, e.T], [e, np.zeros((fixed_idx.size, fixed_idx.size))]])
    rhs = np.concatenate([2.0 * w * y, y[fixed_idx]])
    return np.linalg.solve(kkt, rhs)[:n]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261010)


def dense_divided_difference(t: np.ndarray, order: int) -> np.ndarray:
    """Разделённые разности по Эйлерсу, плотно и в лоб: D_k = diag(k / (t_{i+k} - t_i)) · ΔD_{k-1}."""
    d = np.eye(t.size)
    for k in range(1, order + 1):
        d = np.diag(k / (t[k:] - t[:-k])) @ np.diff(d, axis=0)
    return d


def reference_general(y, lam, fidelity, fixed, penalty_dense):
    """Минимум Σ f_i (y_i - z_i)² + λ zᵀPz при z_K = y_K через систему ККТ."""
    n = y.size
    h = 2.0 * (np.diag(np.where(fixed, 0.0, fidelity)) + lam * penalty_dense)
    idx = np.flatnonzero(fixed)
    e = np.zeros((idx.size, n))
    e[np.arange(idx.size), idx] = 1.0
    kkt = np.block([[h, e.T], [e, np.zeros((idx.size, idx.size))]])
    rhs = np.concatenate([2.0 * np.where(fixed, 0.0, fidelity) * y, y[idx]])
    return np.linalg.solve(kkt, rhs)[:n]


def irregular_times(rng: np.random.Generator, n: int) -> np.ndarray:
    """Моменты с неравномерным шагом от 0.5 до 5 (как у телеметрии гипервизора)."""
    return np.cumsum(rng.uniform(0.5, 5.0, size=n))
