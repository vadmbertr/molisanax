"""Tests for solver.py: Euler, Heun, solve_ode, solve_sde."""

import jax
import jax.numpy as jnp
import pytest

from molisanax.solver import Euler, Heun, solve_ode, solve_sde


# Uniform velocity field: dlat/dt = dlat_rate, dlon/dt = dlon_rate
def uniform_term(dlat_rate, dlon_rate):
    def term(t, y, args):
        return jnp.array([dlat_rate, dlon_rate])
    return term


class TestSolverStep:
    """Unit tests for single-step methods."""

    def test_euler_step(self):
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(1.0)
        term = uniform_term(0.1, 0.2)
        y1 = solver.step(term, jnp.array(0.0), y0, dt, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_step_constant_field(self):
        # For constant field, Heun and Euler agree exactly
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(1.0)
        term = uniform_term(0.1, 0.2)
        y1 = solver.step(term, jnp.array(0.0), y0, dt, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))


class TestSolveODE:
    """Integration tests for solve_ode."""

    def _uniform_ts(self, n_steps=10, dt=3600.0):
        return jnp.linspace(0.0, n_steps * dt, n_steps + 1)

    def test_uniform_field_euler(self):
        dlat = 1e-4  # deg/s
        dlon = 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts = self._uniform_ts(n_steps=100, dt=10.0)
        T = ts[-1] - ts[0]
        traj = solve_ode(uniform_term(dlat, dlon), None, y0, ts, Euler())
        assert traj.shape == (len(ts), 2)
        assert traj[0, 0] == pytest.approx(10.0)
        expected_lat = y0[0] + dlat * T
        expected_lon = y0[1] + dlon * T
        assert float(traj[-1, 0]) == pytest.approx(expected_lat, rel=1e-4)
        assert float(traj[-1, 1]) == pytest.approx(expected_lon, rel=1e-4)

    def test_uniform_field_heun(self):
        dlat = 1e-4
        dlon = 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts = self._uniform_ts(n_steps=100, dt=10.0)
        T = ts[-1] - ts[0]
        traj = solve_ode(uniform_term(dlat, dlon), None, y0, ts, Heun())
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * float(T), rel=1e-5)

    def test_first_point_equals_y0(self):
        y0 = jnp.array([5.0, 10.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        traj = solve_ode(uniform_term(0.0, 0.0), None, y0, ts)
        assert jnp.allclose(traj[0], y0)

    def test_static_field_no_motion(self):
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        traj = solve_ode(uniform_term(0.0, 0.0), None, y0, ts)
        assert jnp.allclose(traj, jnp.zeros_like(traj))

    def test_jit_compatible(self):
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_term(1e-4, 2e-4)
        jitted = jax.jit(lambda y: solve_ode(term, None, y, ts))
        traj = jitted(y0)
        assert traj.shape == (11, 2)

    def test_reverse_mode_grad(self):
        """Gradient of final lat w.r.t. initial lat must be 1 (advection-only)."""
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_term(1e-4, 0.0)

        def loss(y0_):
            traj = solve_ode(term, None, y0_, ts)
            return traj[-1, 0]

        g = jax.grad(loss)(y0)
        # dlat_final / dlat_initial = 1 for pure translation
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        # dlat_final / dlon_initial = 0
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    def test_forward_mode_jvp(self):
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_term(1e-4, 2e-4)

        def fn(y0_):
            return solve_ode(term, None, y0_, ts)

        _, tangent = jax.jvp(fn, (y0,), (jnp.ones(2),))
        assert jnp.all(jnp.isfinite(tangent))


class TestSolveSDE:
    """Tests for solve_sde."""

    def test_shape(self):
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        key = jax.random.key(0)

        def drift(t, y, args):
            return jnp.array([1e-4, 2e-4])

        def diffusion(t, y, args):
            return jnp.zeros((2, 2))

        traj = solve_sde(drift, diffusion, None, y0, ts, key, n_samples=5, n_noise=2)
        assert traj.shape == (5, 11, 2)

    def test_zero_diffusion_matches_ode(self):
        """With zero diffusion, all SDE samples should match the ODE trajectory."""
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        key = jax.random.key(42)

        def drift(t, y, args):
            return jnp.array([1e-4, 2e-4])

        def diffusion(t, y, args):
            return jnp.zeros((2, 2))

        sde_traj = solve_sde(drift, diffusion, None, y0, ts, key, n_samples=3, n_noise=2)
        ode_traj = solve_ode(drift, None, y0, ts)
        # All SDE samples should equal the ODE (up to float precision)
        for i in range(3):
            assert jnp.allclose(sde_traj[i], ode_traj, atol=1e-5)

    def test_ensemble_mean_close_to_ode_with_small_noise(self):
        """Mean of large SDE ensemble should be close to ODE solution."""
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 10.0, 11)
        key = jax.random.key(0)
        scale = 1e-6  # very small noise

        def drift(t, y, args):
            return jnp.array([1e-4, 2e-4])

        def diffusion(t, y, args):
            return scale * jnp.eye(2)

        n = 200
        sde_traj = solve_sde(drift, diffusion, None, y0, ts, key, n_samples=n, n_noise=2)
        ode_traj = solve_ode(drift, None, y0, ts)
        mean_sde = sde_traj.mean(axis=0)
        # Ensemble mean within 1e-3 degrees of ODE
        assert jnp.allclose(mean_sde, ode_traj, atol=1e-3)

    def test_jit_compatible(self):
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        key = jax.random.key(0)

        def drift(t, y, args):
            return jnp.zeros(2)

        def diffusion(t, y, args):
            return jnp.zeros((2, 2))

        fn = jax.jit(lambda k: solve_sde(drift, diffusion, None, y0, ts, k, n_samples=3, n_noise=2))
        traj = fn(key)
        assert traj.shape == (3, 11, 2)
