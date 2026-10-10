"""Свойства на случайных входах (hypothesis)."""

import numpy as np
from conftest import reference_recency
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from whittaker_trosna import smooth_with_mobility

values = st.floats(-1e3, 1e3, allow_nan=False)


@st.composite
def problems(draw):
    n = draw(st.integers(1, 40))
    y = draw(arrays(np.float64, n, elements=values))
    c = draw(arrays(np.float64, n, elements=st.sampled_from([0.0, 0.01, 0.3, 0.7, 1.0])))
    lam = draw(st.sampled_from([0.0, 1e-2, 1.0, 50.0, 1e4]))
    order = draw(st.integers(1, 3))
    return y, c, lam, order


@settings(max_examples=300, deadline=None)
@given(problems())
def test_matches_reference_and_respects_fixed_points(problem):
    y, c, lam, order = problem
    z = smooth_with_mobility(y, lam, c, order=order)
    assert z.shape == y.shape
    assert np.all(np.isfinite(z))
    np.testing.assert_array_equal(z[c == 0], y[c == 0])
    if lam > 0 and y.size > order and np.any(c > 0):
        scale = 1.0 + np.max(np.abs(y))
        np.testing.assert_allclose(z, reference_recency(y, lam, c, order), atol=1e-6 * scale)


@settings(max_examples=200, deadline=None)
@given(problems())
def test_output_stays_within_reasonable_bounds(problem):
    """Сглаживание не порождает выбросов: для order=1 результат в пределах [min y, max y]."""
    y, c, lam, _ = problem
    z = smooth_with_mobility(y, lam, c, order=1)
    tol = 1e-9 * (1.0 + np.max(np.abs(y)))
    assert np.all(z >= y.min() - tol)
    assert np.all(z <= y.max() + tol)
