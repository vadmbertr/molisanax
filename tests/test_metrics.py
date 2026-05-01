"""Tests for metrics.py."""

import jax
import jax.numpy as jnp
import pytest

from molisanax.geo import EARTH_RADIUS
from molisanax.metrics import liu_index, normalized_separation_distance, separation_distance


class TestSeparationDistance:
    def test_identical_trajectories(self):
        y = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        dist = separation_distance(y, y)
        assert jnp.allclose(dist, jnp.zeros(3), atol=1e-3)

    def test_shape(self):
        y = jnp.ones((10, 2))
        y_ref = jnp.zeros((10, 2))
        dist = separation_distance(y, y_ref)
        assert dist.shape == (10,)

    def test_known_distance(self):
        # Two points 1 degree latitude apart at equator
        y = jnp.array([[0.0, 0.0]])
        y_ref = jnp.array([[1.0, 0.0]])
        dist = separation_distance(y, y_ref)
        expected = EARTH_RADIUS * jnp.radians(1.0)
        assert float(dist[0]) == pytest.approx(expected, rel=1e-4)

    def test_grad_is_finite(self):
        y_ref = jnp.zeros((5, 2))
        y = jnp.ones((5, 2)) * 0.1

        g = jax.grad(lambda traj: separation_distance(traj, y_ref).sum())(y)
        assert jnp.all(jnp.isfinite(g))

    def test_vmap_over_ensemble(self):
        ensemble = jnp.ones((4, 10, 2))
        y_ref = jnp.zeros((10, 2))
        dists = jax.vmap(lambda y: separation_distance(y, y_ref))(ensemble)
        assert dists.shape == (4, 10)


class TestNormalizedSeparationDistance:
    def test_zero_separation_gives_zero(self):
        y = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        nsd = normalized_separation_distance(y, y)
        assert jnp.allclose(nsd, jnp.zeros(3), atol=1e-3)

    def test_shape(self):
        y = jnp.ones((8, 2))
        y_ref = jnp.zeros((8, 2))
        nsd = normalized_separation_distance(y, y_ref)
        assert nsd.shape == (8,)

    def test_at_t0_ref_length_is_zero_so_result_is_zero(self):
        y = jnp.array([[1.0, 0.0], [2.0, 0.0]])
        y_ref = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        nsd = normalized_separation_distance(y, y_ref)
        # At t=0, cumulative ref arc length = 0 → safe_divide returns 0
        assert float(nsd[0]) == pytest.approx(0.0)


class TestLiuIndex:
    def test_identical_trajectories_give_zero(self):
        y = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        li = liu_index(y, y)
        assert jnp.allclose(li, jnp.zeros(3), atol=1e-3)

    def test_shape(self):
        y = jnp.ones((6, 2))
        y_ref = jnp.zeros((6, 2))
        li = liu_index(y, y_ref)
        assert li.shape == (6,)

    def test_increasing_in_time_for_diverging_trajectories(self):
        # y moves northward, y_ref stays still → separation grows → Liu grows
        T = 5
        y = jnp.stack([jnp.arange(T, dtype=float), jnp.zeros(T)], axis=1) * 0.1
        y_ref = jnp.zeros((T, 2))
        li = liu_index(y, y_ref)
        # Liu index should be non-decreasing once both trajectories have moved
        assert jnp.all(li[2:] >= li[1:-1] - 1e-6)

    def test_grad_is_finite(self):
        y_ref = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        y = jnp.array([[0.05, 0.0], [0.15, 0.0], [0.25, 0.0]])
        g = jax.grad(lambda traj: liu_index(traj, y_ref).sum())(y)
        assert jnp.all(jnp.isfinite(g))
