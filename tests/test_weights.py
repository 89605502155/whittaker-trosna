import numpy as np
import pytest

from whittaker_trosna import WEIGHT_FUNCTIONS, normalized_age, recency_coefficients

PARAMS = {
    "linear": [{}],
    "power": [{"p": 0.5}, {"p": 1.0}, {"p": 3.0}],
    "exponential": [{"k": -5.0}, {"k": 1e-15}, {"k": 0.5}, {"k": 8.0}],
    "cosine": [{}],
    "logistic": [{}, {"center": 0.0, "steepness": 3.0}, {"center": 0.9, "steepness": 30.0}],
}
CASES = [(name, p) for name, ps in PARAMS.items() for p in ps]


def test_all_builtin_functions_are_covered():
    assert set(PARAMS) == set(WEIGHT_FUNCTIONS)


def test_normalized_age():
    np.testing.assert_allclose(normalized_age(5), [1.0, 0.75, 0.5, 0.25, 0.0])
    np.testing.assert_array_equal(normalized_age(1), [0.0])
    with pytest.raises(ValueError, match="положительной"):
        normalized_age(0)


@pytest.mark.parametrize(("name", "params"), CASES)
@pytest.mark.parametrize("n", [2, 3, 10, 257])
def test_builtin_endpoints_range_and_monotonicity(name, params, n):
    c = recency_coefficients(n, name, **params)
    assert c[-1] == 0.0, "самая свежая точка должна быть закреплена"
    assert c[0] == pytest.approx(1.0, abs=1e-12), "самая старая точка — полная сила"
    assert np.all((c >= 0) & (c <= 1))
    # от свежих к старым коэффициент не убывает
    assert np.all(np.diff(c[::-1]) >= -1e-15)


def test_single_point_is_fixed():
    for name in WEIGHT_FUNCTIONS:
        np.testing.assert_array_equal(recency_coefficients(1, name), [0.0])


def test_shapes_of_specific_functions():
    t = normalized_age(5)
    np.testing.assert_allclose(recency_coefficients(5, "power", p=2), t**2)
    np.testing.assert_allclose(recency_coefficients(5, "cosine"), [1, 0.853553, 0.5, 0.146447, 0], atol=1e-6)
    # exponential при k → 0 переходит в линейную
    np.testing.assert_allclose(recency_coefficients(5, "exponential", k=1e-9), t, atol=1e-8)
    # положительное k защищает меньше свежих точек, чем линейная функция
    assert np.all(recency_coefficients(5, "exponential", k=5)[1:-1] > t[1:-1])
    assert np.all(recency_coefficients(5, "exponential", k=-5)[1:-1] < t[1:-1])


def test_custom_callable_and_array():
    c = recency_coefficients(4, lambda t, a: a * t, a=0.5)
    np.testing.assert_allclose(c, [0.5, 1 / 3, 1 / 6, 0.0])
    arr = [1.0, 0.0, 0.3, 0.0]
    np.testing.assert_array_equal(recency_coefficients(4, arr), arr)


def test_rounding_noise_at_bounds_is_clipped():
    c = recency_coefficients(3, [1.0 + 1e-14, 0.5, -1e-14])
    np.testing.assert_array_equal(c, [1.0, 0.5, 0.0])


@pytest.mark.parametrize(
    ("weight", "params", "err"),
    [
        ("nope", {}, ValueError),
        ("power", {"p": 0}, ValueError),
        ("power", {"p": -1}, ValueError),
        ("exponential", {"k": np.inf}, ValueError),
        ("logistic", {"center": 1.5}, ValueError),
        ("logistic", {"steepness": 0}, ValueError),
        ([0.1, 0.2], {}, ValueError),
        ([0.1, 0.2, 1.5], {}, ValueError),
        ([0.1, -0.2, 0.0], {}, ValueError),
        ([0.1, np.nan, 0.0], {}, ValueError),
        ([0.1, 0.2, 0.0], {"p": 2}, TypeError),
        (lambda t: t[:-1], {}, ValueError),
    ],
)
def test_invalid_weights(weight, params, err):
    with pytest.raises(err):
        recency_coefficients(3, weight, **params)
