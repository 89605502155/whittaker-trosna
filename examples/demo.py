"""Наглядное сравнение классического метода Уиттекера и варианта со свежестью.

Запуск: ``uv run python examples/demo.py`` — графики сохраняются в ``examples/figures``.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from whittaker_trosna import normalized_age, recency_coefficients, recency_whittaker, whittaker

OUT = Path(__file__).parent / "figures"
WEIGHTS = [
    ("linear", {}, "linear"),
    ("power", {"p": 3.0}, "power, p=3"),
    ("exponential", {"k": 8.0}, "exponential, k=8"),
    ("cosine", {}, "cosine"),
    ("logistic", {}, "logistic"),
]


def plot_weights() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    t = normalized_age(201)
    for name, params, label in WEIGHTS:
        ax.plot(t, recency_coefficients(201, name, **params), label=label)
    ax.set_xlabel("нормированный возраст точки t (0 — самая свежая)")
    ax.set_ylabel("коэффициент подвижности c")
    ax.set_title("Весовые функции свежести")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "weights.png", dpi=130)


def plot_series(name: str, truth: np.ndarray, y: np.ndarray, lam: float, title: str) -> None:
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    curves = [("классический", whittaker(y, lam), {"color": "black", "lw": 2})]
    curves += [
        (label, recency_whittaker(y, lam, weight=w, **p), {"lw": 1.5})
        for w, p, label in WEIGHTS
        if w in ("linear", "power", "exponential")
    ]
    zoom = slice(len(y) - 30, len(y))
    x = np.arange(len(y))
    for a, sl in ((ax, slice(None)), (ax_zoom, zoom)):
        a.plot(x[sl], y[sl], ".", color="0.6", ms=4, label="данные")
        a.plot(x[sl], truth[sl], "--", color="tab:green", lw=1, label="истина")
        for label, z, style in curves:
            a.plot(x[sl], z[sl], label=label, **style)
        a.grid(alpha=0.3)
    ax.set_title(f"{title}, λ = {lam:g}")
    ax_zoom.set_title("последние 30 точек")
    ax_zoom.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    plot_weights()

    n = 200
    truth = np.sin(np.linspace(0, 4 * np.pi, n))
    plot_series("sine", truth, truth + 0.3 * rng.normal(size=n), 300.0, "Зашумлённая синусоида")

    truth = np.r_[np.zeros(185), np.ones(15)]
    plot_series("regime_shift", truth, truth + 0.1 * rng.normal(size=n), 300.0, "Смена уровня в конце ряда")

    truth = 0.02 * np.arange(n) + np.where(np.arange(n) > 170, 0.08 * (np.arange(n) - 170), 0.0)
    plot_series("trend_break", truth, truth + 0.3 * rng.normal(size=n), 300.0, "Излом тренда в конце ряда")
    print(f"графики сохранены в {OUT}")


if __name__ == "__main__":
    main()
