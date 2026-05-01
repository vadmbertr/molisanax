"""Tests for geo.py: constants, safe ops, haversine, unit conversions."""

import jax
import jax.numpy as jnp
import pytest

from molisanax.geo import (
    EARTH_RADIUS,
    degrees_to_meters,
    haversine,
    meters_to_degrees,
    safe_divide,
    safe_log,
    safe_sqrt,
)


def test_earth_radius():
    assert EARTH_RADIUS == pytest.approx(6_371_008.8)


class TestSafeSqrt:
    def test_positive(self):
        assert safe_sqrt(jnp.array(4.0)) == pytest.approx(2.0)

    def test_zero_returns_zero(self):
        assert float(safe_sqrt(jnp.array(0.0))) == 0.0

    def test_negative_returns_zero(self):
        assert float(safe_sqrt(jnp.array(-1.0))) == 0.0

    def test_grad_at_zero_is_finite(self):
        # Must not produce NaN gradient at x=0
        g = jax.grad(lambda x: safe_sqrt(x))(jnp.array(0.0))
        assert jnp.isfinite(g)


class TestSafeLog:
    def test_positive(self):
        assert safe_log(jnp.array(1.0)) == pytest.approx(0.0)

    def test_zero_returns_neginf(self):
        assert jnp.isinf(safe_log(jnp.array(0.0)))

    def test_grad_at_positive_is_finite(self):
        g = jax.grad(lambda x: safe_log(x))(jnp.array(2.0))
        assert jnp.isfinite(g)


class TestSafeDivide:
    def test_normal(self):
        assert safe_divide(jnp.array(6.0), jnp.array(2.0)) == pytest.approx(3.0)

    def test_zero_denom_returns_zero(self):
        assert float(safe_divide(jnp.array(5.0), jnp.array(0.0))) == 0.0

    def test_grad_finite(self):
        g = jax.grad(lambda a: safe_divide(a, jnp.array(2.0)))(jnp.array(3.0))
        assert jnp.isfinite(g)


class TestHaversine:
    def test_same_point_is_zero(self):
        y = jnp.array([48.0, 2.0])
        assert float(haversine(y, y)) == pytest.approx(0.0, abs=1e-3)

    def test_north_pole_to_equator(self):
        # Quarter of Earth circumference ≈ EARTH_RADIUS * pi/2
        north = jnp.array([90.0, 0.0])
        equator = jnp.array([0.0, 0.0])
        expected = EARTH_RADIUS * jnp.pi / 2
        assert float(haversine(north, equator)) == pytest.approx(expected, rel=1e-4)

    def test_equator_one_degree_lon(self):
        # At equator, 1 degree longitude ≈ EARTH_RADIUS * pi/180 metres
        y1 = jnp.array([0.0, 0.0])
        y2 = jnp.array([0.0, 1.0])
        expected = EARTH_RADIUS * jnp.pi / 180
        assert float(haversine(y1, y2)) == pytest.approx(expected, rel=1e-4)

    def test_grad_is_finite_at_non_coincident_points(self):
        y1 = jnp.array([48.0, 2.0])
        y2 = jnp.array([49.0, 3.0])
        g = jax.grad(lambda a: haversine(a, y2))(y1)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_is_finite_at_coincident_points(self):
        y = jnp.array([48.0, 2.0])
        g = jax.grad(lambda a: haversine(a, y))(y)
        assert jnp.all(jnp.isfinite(g))


class TestUnitConversions:
    def test_round_trip_at_equator(self):
        # At equator, conversion should be lossless
        disp_m = jnp.array([1000.0, 1000.0])
        lat = jnp.array(0.0)
        disp_deg = meters_to_degrees(disp_m, lat)
        recovered = degrees_to_meters(disp_deg, lat)
        assert jnp.allclose(recovered, disp_m, rtol=1e-5)

    def test_round_trip_at_45deg(self):
        disp_m = jnp.array([5000.0, 5000.0])
        lat = jnp.array(45.0)
        disp_deg = meters_to_degrees(disp_m, lat)
        recovered = degrees_to_meters(disp_deg, lat)
        assert jnp.allclose(recovered, disp_m, rtol=1e-5)

    def test_longitude_scaling(self):
        # At 60 degrees latitude, 1 degree lon ≈ 0.5 * (1 deg lat in metres)
        lat = jnp.array(60.0)
        one_deg = jnp.array([1.0, 1.0])
        m = degrees_to_meters(one_deg, lat)
        assert m[1] == pytest.approx(m[0] * 0.5, rel=1e-3)
