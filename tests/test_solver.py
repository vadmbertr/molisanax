"""Tests for solver.py: Euler, Heun, unified solve()."""

import jax
import jax.numpy as jnp
import pytest

from molisanax.solver import RK4, Euler, Heun, solve


# ODE term: constant velocity
def uniform_ode_term(dlat, dlon):
    def term(t, y, args):
        return jnp.array([dlat, dlon])
    return term


# SDE term: constant drift + constant noise amplitude, z-aware
def uniform_sde_term(dlat, dlon, noise_scale=1e-6):
    def term(t, y, args, z):
        f = jnp.array([dlat, dlon])
        g = jnp.full(2, noise_scale)
        return f + g * z
    return term


class TestSolverStep:
    def test_euler_ode_step_constant_field(self):
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0, jnp.array(1.0), None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_ode_step_constant_field(self):
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0, jnp.array(1.0), None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_euler_sde_step_zero_noise(self):
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.zeros(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_sde_step_zero_noise(self):
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.zeros(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_sde_uses_same_z_both_stages(self):
        # z is non-zero; confirm the call succeeds and returns a finite result
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=1.0)
        z = jnp.ones(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(0.01), None, z)
        assert jnp.all(jnp.isfinite(y1))

    def test_rk4_ode_step_constant_field(self):
        solver = RK4()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0, jnp.array(1.0), None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_rk4_sde_step_zero_noise(self):
        solver = RK4()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.zeros(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))


class TestSolveODE:
    def _ts(self, n=100, dt=10.0):
        return jnp.linspace(0.0, n * dt, n + 1)

    def test_uniform_field_euler(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts = self._ts()
        T = float(ts[-1] - ts[0])
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, ts, Euler())
        assert traj.shape == (len(ts), 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) + dlon * T, rel=1e-4)

    def test_uniform_field_heun(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts = self._ts()
        T = float(ts[-1] - ts[0])
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, ts, Heun())
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-5)

    def test_first_point_equals_y0(self):
        y0 = jnp.array([5.0, 10.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        traj = solve(uniform_ode_term(0.0, 0.0), None, y0, ts)
        assert jnp.allclose(traj[0], y0)

    def test_no_motion(self):
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        traj = solve(uniform_ode_term(0.0, 0.0), None, y0, ts)
        assert jnp.allclose(traj, jnp.zeros_like(traj))

    def test_jit_compatible(self):
        y0 = jnp.array([0.0, 0.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_ode_term(1e-4, 2e-4)
        traj = jax.jit(lambda y: solve(term, None, y, ts))(y0)
        assert traj.shape == (11, 2)

    def test_reverse_mode_grad(self):
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, None, y0_, ts)[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    def test_forward_mode_jvp(self):
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_ode_term(1e-4, 2e-4)
        _, tangent = jax.jvp(lambda y: solve(term, None, y, ts), (y0,), (jnp.ones(2),))
        assert jnp.all(jnp.isfinite(tangent))

    def test_requires_no_key(self):
        y0 = jnp.zeros(2)
        ts = jnp.linspace(0.0, 10.0, 5)
        traj = solve(uniform_ode_term(0.0, 0.0), None, y0, ts)
        assert traj.shape == (5, 2)

    def test_uniform_field_rk4(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts = self._ts()
        T = float(ts[-1] - ts[0])
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, ts, RK4())
        assert traj.shape == (len(ts), 2)
        # RK4 is exact for constant velocity (up to float32 accumulation).
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) + dlon * T, rel=1e-4)

    def test_rk4_convergence_order_on_linear_field(self):
        # Velocity grows linearly with lat: v_lat(t, y) = alpha * y[0].
        # Exact solution: y(T) = y0 * exp(alpha * T). RK4 should hit 4th-order
        # accuracy and clearly outperform Heun under step refinement.
        # Use float64 so the truncation error dominates over rounding error.
        from jax import config

        config.update("jax_enable_x64", True)
        try:
            alpha = 0.1

            def term(t, y, args):
                return jnp.array([alpha * y[0], 0.0], dtype=jnp.float64)

            y0 = jnp.array([1.0, 0.0], dtype=jnp.float64)
            T = 1.0
            exact = float(y0[0]) * float(jnp.exp(alpha * T))

            def err(solver, n):
                ts = jnp.linspace(0.0, T, n + 1, dtype=jnp.float64)
                return abs(float(solve(term, None, y0, ts, solver)[-1, 0]) - exact)

            err_rk4_coarse = err(RK4(), 4)
            err_rk4_fine   = err(RK4(), 8)
            err_heun_coarse = err(Heun(), 4)

            # RK4 reduces error by ~16x when dt is halved (4th order).
            assert err_rk4_coarse / max(err_rk4_fine, 1e-30) > 8.0
            # And RK4 is strictly more accurate than Heun at the same step count.
            assert err_rk4_coarse < err_heun_coarse
        finally:
            config.update("jax_enable_x64", False)

    def test_rk4_reverse_mode_grad(self):
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(0.0, 100.0, 11)
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, None, y0_, ts, RK4())[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    # --- backwards-in-time ---

    def test_backwards_in_time_constant_field(self):
        # Integrating backwards from y0 with constant velocity over time T
        # produces y0 - v*T.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(100.0, 0.0, 101)  # strictly decreasing
        T = float(ts[0] - ts[-1])
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, ts, Heun())
        assert traj.shape == (len(ts), 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) - dlat * T, rel=1e-5)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) - dlon * T, rel=1e-5)

    def test_forward_then_backward_returns_to_origin(self):
        # Run forward to T, then backward from the end point to 0: should
        # land back at y0 to within solver accuracy.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        ts_fwd = jnp.linspace(0.0, 100.0, 51)
        ts_bwd = jnp.linspace(100.0, 0.0, 51)
        term = uniform_ode_term(dlat, dlon)
        fwd = solve(term, None, y0, ts_fwd, RK4())
        bwd = solve(term, None, fwd[-1], ts_bwd, RK4())
        assert jnp.allclose(bwd[-1], y0, atol=1e-8)

    def test_backwards_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        ts = jnp.linspace(50.0, 0.0, 11)
        term = uniform_ode_term(1e-4, 0.0)
        traj = jax.jit(lambda y: solve(term, None, y, ts))(y0)
        assert traj.shape == (11, 2)

        def loss(y0_):
            return solve(term, None, y0_, ts)[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)


class TestSolveSDE:
    def _ts(self, n=10, dt=10.0):
        return jnp.linspace(0.0, n * dt, n + 1)

    def test_single_sample_shape(self):
        y0 = jnp.zeros(2)
        ts = self._ts()
        key = jax.random.key(0)
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, key=key, n_noise=2)
        assert traj.shape == (len(ts), 2)

    def test_ensemble_shape(self):
        y0 = jnp.zeros(2)
        ts = self._ts()
        key = jax.random.key(0)
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, key=key, n_noise=2, n_samples=7)
        assert traj.shape == (7, len(ts), 2)

    def test_zero_noise_matches_ode(self):
        y0 = jnp.array([10.0, 20.0])
        ts = self._ts(n=50)
        key = jax.random.key(0)
        # noise_scale=0.0 → g*z = 0 regardless of z → matches ODE
        sde_traj = solve(uniform_sde_term(1e-4, 2e-4, noise_scale=0.0), None, y0, ts,
                         key=key, n_noise=2)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), None, y0, ts)
        assert jnp.allclose(sde_traj, ode_traj, atol=1e-5)

    def test_ensemble_mean_close_to_ode_with_small_noise(self):
        y0 = jnp.zeros(2)
        ts = self._ts(n=10)
        key = jax.random.key(0)
        ensemble = solve(uniform_sde_term(1e-4, 2e-4, noise_scale=1e-8), None, y0, ts,
                         key=key, n_noise=2, n_samples=200)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), None, y0, ts)
        assert jnp.allclose(ensemble.mean(axis=0), ode_traj, atol=1e-4)

    def test_missing_key_and_noise_raises(self):
        # n_noise without key or noise → SDE mode but no noise source
        y0 = jnp.zeros(2)
        ts = self._ts()
        with pytest.raises(ValueError):
            solve(uniform_sde_term(0.0, 0.0), None, y0, ts, n_noise=2)

    def test_missing_n_noise_with_key_raises(self):
        # key without n_noise → cannot auto-sample without knowing latent dim
        y0 = jnp.zeros(2)
        ts = self._ts()
        with pytest.raises(ValueError, match="n_noise"):
            solve(uniform_sde_term(0.0, 0.0), None, y0, ts, key=jax.random.key(0))

    def test_jit_compatible(self):
        y0 = jnp.zeros(2)
        ts = self._ts()
        term = uniform_sde_term(1e-4, 2e-4)
        fn = jax.jit(lambda k: solve(term, None, y0, ts, key=k, n_noise=2, n_samples=3))
        traj = fn(jax.random.key(0))
        assert traj.shape == (3, len(ts), 2)

    def test_custom_n_noise(self):
        # term uses z of dimension 4 (e.g., a generative model)
        def term_4d(t, y, args, z):
            return jnp.array([z[0] + z[1], z[2] + z[3]]) * 1e-6
        y0 = jnp.zeros(2)
        ts = self._ts()
        traj = solve(term_4d, None, y0, ts, key=jax.random.key(0), n_noise=4)
        assert traj.shape == (len(ts), 2)
        assert jnp.all(jnp.isfinite(traj))

    # --- pre-sampled noise ---

    def test_presampled_noise_single_shape(self):
        y0 = jnp.zeros(2)
        ts = self._ts()
        n_steps = len(ts) - 1
        noise = jnp.zeros((n_steps, 2))
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, noise=noise)
        assert traj.shape == (len(ts), 2)

    def test_presampled_noise_single_zero_matches_ode(self):
        y0 = jnp.array([10.0, 20.0])
        ts = self._ts(n=50)
        n_steps = len(ts) - 1
        noise = jnp.zeros((n_steps, 2))
        sde_traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, noise=noise)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), None, y0, ts)
        assert jnp.allclose(sde_traj, ode_traj, atol=1e-5)

    def test_presampled_noise_ensemble_shape(self):
        y0 = jnp.zeros(2)
        ts = self._ts()
        n_steps = len(ts) - 1
        noise = jnp.zeros((5, n_steps, 2))
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, noise=noise)
        assert traj.shape == (5, len(ts), 2)

    def test_presampled_noise_no_key_needed(self):
        y0 = jnp.zeros(2)
        ts = self._ts()
        n_steps = len(ts) - 1
        noise = jax.random.normal(jax.random.key(7), (n_steps, 2))
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, noise=noise)
        assert jnp.all(jnp.isfinite(traj))

    def test_presampled_vs_autosampled_same_noise_same_traj(self):
        # Reproduce the exact noise that auto-sample uses, then compare.
        y0 = jnp.array([10.0, 20.0])
        ts = self._ts(n=20)
        key = jax.random.key(42)
        n_steps = len(ts) - 1

        traj_auto = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, key=key, n_noise=2)

        # Auto-sample draws (1, n_steps, n_noise)[0]
        noise_repro = jax.random.normal(key, shape=(1, n_steps, 2), dtype=y0.dtype)[0]
        traj_noise = solve(uniform_sde_term(1e-4, 2e-4), None, y0, ts, noise=noise_repro)

        assert jnp.allclose(traj_auto, traj_noise, atol=1e-6)
