"""Tests for interpolation.py."""

import jax
import jax.numpy as jnp
import pytest

from molisanax.interpolation import bilinear_interp_2d, linear_interp_1d, spatiotemporal_interp


class TestLinearInterp1D:
    def test_at_node(self):
        coords = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([10.0, 20.0, 30.0])
        assert float(linear_interp_1d(values, coords, jnp.array(1.0))) == pytest.approx(20.0)

    def test_midpoint(self):
        coords = jnp.array([0.0, 2.0])
        values = jnp.array([0.0, 4.0])
        assert float(linear_interp_1d(values, coords, jnp.array(1.0))) == pytest.approx(2.0)

    def test_grad_is_finite(self):
        coords = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, 3.0, 6.0])
        g = jax.grad(lambda x: linear_interp_1d(values, coords, x))(jnp.array(0.7))
        assert jnp.isfinite(g)


class TestBilinearInterp2D:
    def setup_method(self):
        # 3x3 grid, values = lat + lon
        self.lats = jnp.array([0.0, 1.0, 2.0])
        self.lons = jnp.array([0.0, 1.0, 2.0])
        self.values = jnp.array(
            [[0.0, 1.0, 2.0],
             [1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0]]
        )

    def test_at_node(self):
        v = bilinear_interp_2d(self.values, self.lats, self.lons, jnp.array(1.0), jnp.array(1.0))
        assert float(v) == pytest.approx(2.0)

    def test_midpoint(self):
        v = bilinear_interp_2d(self.values, self.lats, self.lons, jnp.array(0.5), jnp.array(0.5))
        assert float(v) == pytest.approx(1.0)  # 0.5 + 0.5

    def test_grad_is_finite(self):
        g = jax.grad(
            lambda lat: bilinear_interp_2d(self.values, self.lats, self.lons, lat, jnp.array(0.5))
        )(jnp.array(0.3))
        assert jnp.isfinite(g)

    def test_jit_compatible(self):
        fn = jax.jit(bilinear_interp_2d, static_argnums=())
        v = fn(self.values, self.lats, self.lons, jnp.array(0.5), jnp.array(1.5))
        assert jnp.isfinite(v)


class TestSpatiotemporalInterp:
    def setup_method(self):
        # 2 time steps, 3x3 spatial grid, values = t + lat + lon
        self.t_coords = jnp.array([0.0, 1.0])
        self.lats = jnp.array([0.0, 1.0, 2.0])
        self.lons = jnp.array([0.0, 1.0, 2.0])
        self.values = jnp.stack([
            jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]),
        ])

    def test_at_node(self):
        v = spatiotemporal_interp(
            self.values, self.t_coords, self.lats, self.lons,
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
        )
        assert float(v) == pytest.approx(0.0)

    def test_mid_time(self):
        v = spatiotemporal_interp(
            self.values, self.t_coords, self.lats, self.lons,
            jnp.array(0.5), jnp.array(1.0), jnp.array(1.0),
        )
        assert float(v) == pytest.approx(2.5)  # 0+1+1=2 at t=0, 1+1+1=3 at t=1 → 2.5

    def test_grad_wrt_lat(self):
        g = jax.grad(
            lambda lat: spatiotemporal_interp(
                self.values, self.t_coords, self.lats, self.lons,
                jnp.array(0.5), lat, jnp.array(1.0),
            )
        )(jnp.array(0.5))
        assert jnp.isfinite(g)
