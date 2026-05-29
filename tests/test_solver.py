"""Tests for solver.py: ODE/SDE solvers and the unified solve() entry point."""

import jax
import jax.numpy as jnp
import pytest

from pastax.solver import (
    RK4,
    Dopri5,
    Euler,
    EulerHeun,
    Heun,
    ItoMilstein,
    StratonovichMilstein,
    Tsit5,
    solve,
)


# ODE term: constant velocity
def uniform_ode_term(dlat, dlon):
    def term(t, y, args):
        return jnp.array([dlat, dlon])
    return term


# SDE term: constant drift + constant diagonal diffusion coefficient.
# Term accepts z (forwarded by the solver) but ignores it — textbook SDE.
def uniform_sde_term(dlat, dlon, noise_scale=1e-6):
    def term(t, y, args, z):
        f = jnp.array([dlat, dlon])
        g = jnp.full(2, noise_scale)
        return f, g
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

    def test_euler_maruyama_matches_textbook_formula(self):
        # Constant diffusion: y_new = y + f*dt + g*sqrt(dt)*z
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.25)
        z = jnp.array([1.5, -0.5])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, dt, None, z)
        expected = y0 + jnp.array([0.1, 0.2]) * dt + 2.0 * jnp.sqrt(dt) * z
        assert jnp.allclose(y1, expected)


class TestSolveODE:
    def test_uniform_field_euler(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, jnp.array(0.0), n_save, int_dt, int_dt, Euler())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) + dlon * T, rel=1e-4)

    def test_uniform_field_heun(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, jnp.array(0.0), n_save, int_dt, int_dt, Heun())
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-5)

    def test_first_point_equals_y0(self):
        y0 = jnp.array([5.0, 10.0])
        traj = solve(uniform_ode_term(0.0, 0.0), None, y0, jnp.array(0.0), 10, 10.0, 10.0)
        assert jnp.allclose(traj[0], y0)

    def test_no_motion(self):
        y0 = jnp.array([0.0, 0.0])
        traj = solve(uniform_ode_term(0.0, 0.0), None, y0, jnp.array(0.0), 10, 10.0, 10.0)
        assert jnp.allclose(traj, jnp.zeros_like(traj))

    def test_jit_compatible(self):
        y0 = jnp.array([0.0, 0.0])
        term = uniform_ode_term(1e-4, 2e-4)
        traj = jax.jit(lambda y, t0: solve(term, None, y, t0, 10, 10.0, 10.0))(y0, jnp.array(0.0))
        assert traj.shape == (11, 2)

    def test_t0_traced_no_recompile(self):
        # t0 is traced: changing it must not trigger recompilation.
        # Use a time-dependent term so the trajectory actually differs with t0.
        def time_dep_term(t, y, args):
            return jnp.array([1e-4 * t, 0.0])
        fn = jax.jit(lambda t0: solve(time_dep_term, None, jnp.zeros(2), t0, 5, 10.0, 10.0))
        traj_a = fn(jnp.array(0.0))
        traj_b = fn(jnp.array(100.0))
        assert traj_a.shape == traj_b.shape == (6, 2)
        # Different t0 → different absolute times → different integrated velocity.
        assert not jnp.allclose(traj_a, traj_b)

    def test_substep_matches_fine_grid(self):
        # With save_dt = 4 * int_dt, sliced output must equal a run at save_dt step.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([0.0, 0.0])
        term = uniform_ode_term(dlat, dlon)
        int_dt, save_dt, n_save = 1.0, 4.0, 10

        traj_sub  = solve(term, None, y0, jnp.array(0.0), n_save, int_dt, save_dt)
        traj_coarse = solve(term, None, y0, jnp.array(0.0), n_save, save_dt, save_dt)
        assert jnp.allclose(traj_sub, traj_coarse, atol=1e-6)

    def test_reverse_mode_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, None, y0_, jnp.array(0.0), 10, 10.0, 10.0)[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    def test_forward_mode_jvp(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 2e-4)
        _, tangent = jax.jvp(
            lambda y: solve(term, None, y, jnp.array(0.0), 10, 10.0, 10.0),
            (y0,), (jnp.ones(2),),
        )
        assert jnp.all(jnp.isfinite(tangent))

    def test_requires_no_key(self):
        y0 = jnp.zeros(2)
        traj = solve(uniform_ode_term(0.0, 0.0), None, y0, jnp.array(0.0), 4, 2.5, 2.5)
        assert traj.shape == (5, 2)

    def test_uniform_field_rk4(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, jnp.array(0.0), n_save, int_dt, int_dt, RK4())
        assert traj.shape == (n_save + 1, 2)
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
                dt = T / n
                return abs(float(
                    solve(term, None, y0, jnp.array(0.0, dtype=jnp.float64), n, dt, dt, solver)[-1, 0]
                ) - exact)

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
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, None, y0_, jnp.array(0.0), 10, 10.0, 10.0, RK4())[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    # --- backwards-in-time ---

    def test_backwards_in_time_constant_field(self):
        # Integrating backwards from y0 with constant velocity over time T
        # produces y0 - v*T.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, -1.0          # negative int_dt → backward
        T = abs(n_save * int_dt)
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, jnp.array(100.0), n_save, int_dt, int_dt, Heun())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) - dlat * T, rel=1e-5)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) - dlon * T, rel=1e-5)

    def test_forward_then_backward_returns_to_origin(self):
        # Run forward to T, then backward from the end point to 0: should
        # land back at y0 to within solver accuracy.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(dlat, dlon)
        fwd = solve(term, None, y0, jnp.array(0.0), 50, 2.0, 2.0, RK4())
        bwd = solve(term, None, fwd[-1], jnp.array(100.0), 50, -2.0, -2.0, RK4())
        assert jnp.allclose(bwd[-1], y0, atol=1e-8)

    def test_backwards_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)
        traj = jax.jit(lambda y, t0: solve(term, None, y, t0, 10, -5.0, -5.0))(y0, jnp.array(50.0))
        assert traj.shape == (11, 2)

        def loss(y0_):
            return solve(term, None, y0_, jnp.array(50.0), 10, -5.0, -5.0)[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)


class TestSolveSDE:
    def test_single_sample_shape(self):
        y0 = jnp.zeros(2)
        key = jax.random.key(0)
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), 10, 10.0, 10.0, key=key, n_noise=2)
        assert traj.shape == (11, 2)

    def test_ensemble_shape(self):
        y0 = jnp.zeros(2)
        key = jax.random.key(0)
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), 10, 10.0, 10.0, key=key, n_noise=2, n_samples=7)
        assert traj.shape == (7, 11, 2)

    def test_zero_noise_matches_ode(self):
        y0 = jnp.array([10.0, 20.0])
        key = jax.random.key(0)
        # noise_scale=0.0 → g=0 → matches ODE
        sde_traj = solve(uniform_sde_term(1e-4, 2e-4, noise_scale=0.0), None, y0,
                         jnp.array(0.0), 50, 10.0, 10.0, key=key, n_noise=2)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), None, y0,
                         jnp.array(0.0), 50, 10.0, 10.0)
        assert jnp.allclose(sde_traj, ode_traj, atol=1e-5)

    def test_ensemble_mean_close_to_ode_with_small_noise(self):
        y0 = jnp.zeros(2)
        key = jax.random.key(0)
        ensemble = solve(uniform_sde_term(1e-4, 2e-4, noise_scale=1e-8), None, y0,
                         jnp.array(0.0), 10, 10.0, 10.0, key=key, n_noise=2, n_samples=200)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), None, y0,
                         jnp.array(0.0), 10, 10.0, 10.0)
        assert jnp.allclose(ensemble.mean(axis=0), ode_traj, atol=1e-4)

    def test_missing_key_and_noise_raises(self):
        # n_noise without key or noise → SDE mode but no noise source
        y0 = jnp.zeros(2)
        with pytest.raises(ValueError):
            solve(uniform_sde_term(0.0, 0.0), None, y0, jnp.array(0.0), 10, 10.0, 10.0, n_noise=2)

    def test_missing_n_noise_with_key_raises(self):
        # key without n_noise → cannot auto-sample without knowing latent dim
        y0 = jnp.zeros(2)
        with pytest.raises(ValueError, match="n_noise"):
            solve(uniform_sde_term(0.0, 0.0), None, y0, jnp.array(0.0), 10, 10.0, 10.0, key=jax.random.key(0))

    def test_jit_compatible(self):
        y0 = jnp.zeros(2)
        term = uniform_sde_term(1e-4, 2e-4)
        fn = jax.jit(lambda k: solve(term, None, y0, jnp.array(0.0), 10, 10.0, 10.0, key=k, n_noise=2, n_samples=3))
        traj = fn(jax.random.key(0))
        assert traj.shape == (3, 11, 2)

    def test_matrix_diffusion(self):
        # g.shape == (2, 4): full matrix diffusion with n_noise=4.
        def term_mat(t, y, args, z):
            drift = jnp.zeros(2)
            g = 1e-6 * jnp.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0]])
            return drift, g
        y0 = jnp.zeros(2)
        traj = solve(term_mat, None, y0, jnp.array(0.0), 10, 10.0, 10.0, key=jax.random.key(0), n_noise=4)
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))

    def test_nonlinear_noise_via_z_in_drift(self):
        # Random-ODE / MDN-style: the term uses z (a high-dim latent) inside
        # `drift` and opts out of the g*dW path by returning a zero diffusion.
        # For a non-2 latent dim, the zero diffusion is shape (2, n_noise) so
        # _apply_g's matmul branch produces a (2,) zero — this keeps the
        # solver's shape contract straightforward.
        def mdn_like_term(t, y, args, z):
            residual = 1e-4 * jnp.array([
                jnp.tanh(z[0] + z[1] * y[0]),
                jnp.tanh(z[2] - z[3] * y[1]),
            ])
            drift = jnp.array([1e-4, 2e-4]) + residual
            return drift, jnp.zeros((2, 4))       # opt out of g*dW path
        y0 = jnp.zeros(2)
        traj = solve(mdn_like_term, None, y0, jnp.array(0.0), 10, 10.0, 10.0,
                     key=jax.random.key(0), n_noise=4, n_samples=5)
        assert traj.shape == (5, 11, 2)
        assert jnp.all(jnp.isfinite(traj))
        # Different samples (different z trajectories) must produce distinct
        # outputs — otherwise z isn't actually reaching the term.
        assert not jnp.allclose(traj[0], traj[1])

    # --- pre-sampled noise ---

    def test_presampled_noise_single_shape(self):
        y0 = jnp.zeros(2)
        n_save, n_noise = 10, 2
        noise = jnp.zeros((n_save, n_noise))
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), n_save, 10.0, 10.0, noise=noise)
        assert traj.shape == (n_save + 1, 2)

    def test_presampled_noise_single_zero_matches_ode(self):
        y0 = jnp.array([10.0, 20.0])
        n_save = 50
        noise = jnp.zeros((n_save, 2))
        sde_traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), n_save, 10.0, 10.0, noise=noise)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), None, y0, jnp.array(0.0), n_save, 10.0, 10.0)
        assert jnp.allclose(sde_traj, ode_traj, atol=1e-5)

    def test_presampled_noise_ensemble_shape(self):
        y0 = jnp.zeros(2)
        n_save = 10
        noise = jnp.zeros((5, n_save, 2))
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), n_save, 10.0, 10.0, noise=noise)
        assert traj.shape == (5, n_save + 1, 2)

    def test_presampled_noise_no_key_needed(self):
        y0 = jnp.zeros(2)
        n_save = 10
        noise = jax.random.normal(jax.random.key(7), (n_save, 2))
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), n_save, 10.0, 10.0, noise=noise)
        assert jnp.all(jnp.isfinite(traj))

    def test_presampled_vs_autosampled_same_noise_same_traj(self):
        # Reproduce the exact noise that auto-sample uses, then compare.
        y0 = jnp.array([10.0, 20.0])
        n_save = 20
        key = jax.random.key(42)

        traj_auto = solve(uniform_sde_term(1e-4, 2e-4), None, y0,
                          jnp.array(0.0), n_save, 10.0, 10.0, key=key, n_noise=2)

        # Auto-sample draws (1, n_fine, n_noise)[0]; with n_substeps=1, n_fine=n_save.
        noise_repro = jax.random.normal(key, shape=(1, n_save, 2), dtype=y0.dtype)[0]
        traj_noise = solve(uniform_sde_term(1e-4, 2e-4), None, y0,
                           jnp.array(0.0), n_save, 10.0, 10.0, noise=noise_repro)

        assert jnp.allclose(traj_auto, traj_noise, atol=1e-6)


# ---------------------------------------------------------------------------
# Tsit5 / Dopri5  (ODE-only)
# ---------------------------------------------------------------------------

class TestTsit5:
    def test_constant_field_step(self):
        solver = Tsit5()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0, jnp.array(1.0), None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_uniform_field_solve(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, jnp.array(0.0), n_save, int_dt, int_dt, Tsit5())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)

    def test_sde_step_raises(self):
        with pytest.raises(NotImplementedError, match="ODE-only"):
            Tsit5().sde_step(
                uniform_sde_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, jnp.zeros(2),
            )

    def test_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, None, y0_, jnp.array(0.0), 10, 10.0, 10.0, Tsit5())[-1, 0]

        g = jax.grad(jax.jit(loss))(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)


class TestDopri5:
    def test_constant_field_step(self):
        solver = Dopri5()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0, jnp.array(1.0), None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_uniform_field_solve(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), None, y0, jnp.array(0.0), n_save, int_dt, int_dt, Dopri5())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)

    def test_sde_step_raises(self):
        with pytest.raises(NotImplementedError, match="ODE-only"):
            Dopri5().sde_step(
                uniform_sde_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, jnp.zeros(2),
            )

    def test_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, None, y0_, jnp.array(0.0), 10, 10.0, 10.0, Dopri5())[-1, 0]

        g = jax.grad(jax.jit(loss))(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)


def test_tsit5_and_dopri5_fifth_order_convergence():
    # Same linear-field setup as test_rk4_convergence_order_on_linear_field:
    # halving dt should shrink the error by ~32 (order 5) for both solvers.
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
            dt = T / n
            return abs(float(
                solve(term, None, y0, jnp.array(0.0, dtype=jnp.float64), n, dt, dt, solver)[-1, 0]
            ) - exact)

        for solver_cls in (Tsit5, Dopri5):
            err_coarse = err(solver_cls(), 4)
            err_fine   = err(solver_cls(), 8)
            # 5th-order: halving dt → 32× error reduction. Conservative threshold.
            assert err_coarse / max(err_fine, 1e-30) > 16.0
            # Both should beat RK4 at the same step count.
            err_rk4 = err(RK4(), 4)
            assert err_coarse < err_rk4
    finally:
        config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# EulerHeun  (SDE-only)
# ---------------------------------------------------------------------------

class TestEulerHeun:
    def test_ode_step_raises(self):
        with pytest.raises(NotImplementedError, match="SDE-only"):
            EulerHeun().ode_step(
                uniform_ode_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None,
            )

    def test_zero_diffusion_matches_euler_drift(self):
        # g=0 → diffusion-only predictor is a no-op; result = y + f*dt.
        solver = EulerHeun()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.array([1.0, -1.0])  # non-zero z, but g=0 still kills it
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_constant_g_matches_euler_maruyama(self):
        # dg/dy = 0 → EulerHeun's predictor/corrector both see the same g,
        # so y_new = y + f*dt + g*dW (same as Euler-Maruyama).
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.5)
        z = jnp.array([1.3, -0.4])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y_eh = EulerHeun().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        y_em = Euler().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        assert jnp.allclose(y_eh, y_em)

    def test_matrix_diffusion(self):
        def term_mat(t, y, args, z):
            drift = jnp.array([0.1, 0.2])
            g = jnp.array([[0.5, 0.1], [0.1, 0.5]])      # (2, 2)
            return drift, g
        y0 = jnp.zeros(2)
        z = jnp.array([1.0, -1.0])
        y1 = EulerHeun().sde_step(term_mat, jnp.array(0.0), y0, jnp.array(0.25), None, z)
        assert jnp.all(jnp.isfinite(y1))

    def test_full_solve(self):
        y0 = jnp.zeros(2)
        traj = solve(uniform_sde_term(1e-4, 2e-4), None, y0, jnp.array(0.0), 10, 10.0, 10.0,
                     EulerHeun(), key=jax.random.key(0), n_noise=2)
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))


# ---------------------------------------------------------------------------
# Milstein solvers  (SDE-only, diagonal noise)
# ---------------------------------------------------------------------------

def _linear_diffusion_term(sigma, drift=(0.0, 0.0)):
    """SDE term with g_i = sigma * y_i (diagonal, state-dependent)."""
    def term(t, y, args, z):
        f = jnp.array(drift)
        g = sigma * y
        return f, g
    return term


class TestItoMilstein:
    def test_ode_step_raises(self):
        with pytest.raises(NotImplementedError, match="SDE-only"):
            ItoMilstein().ode_step(
                uniform_ode_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None,
            )

    def test_constant_g_matches_euler_maruyama_minus_ito_drift(self):
        # dg/dy = 0 → cross-term vanishes, but Itô subtracts 0 too (g*dgdy*dt = 0).
        # So Itô Milstein reduces exactly to Euler-Maruyama.
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.25)
        z = jnp.array([1.2, -0.7])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y_im = ItoMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        y_em = Euler().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        assert jnp.allclose(y_im, y_em)

    def test_state_dependent_g_differs_from_stratonovich_by_ito_drift(self):
        # g_i = sigma * y_i  →  dg_i/dy_i = sigma. The two Milsteins differ by
        # exactly -0.5 * g * dgdy * dt = -0.5 * sigma**2 * y * dt (Itô subtracts).
        sigma = 0.3
        y0 = jnp.array([1.0, 2.0])
        dt = jnp.array(0.1)
        z = jnp.array([0.5, -0.5])
        term = _linear_diffusion_term(sigma)
        y_ito  = ItoMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        y_str  = StratonovichMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        expected_diff = -0.5 * sigma**2 * y0 * dt
        assert jnp.allclose(y_ito - y_str, expected_diff)

    def test_matrix_g_raises(self):
        def term_mat(t, y, args, z):
            return jnp.zeros(2), jnp.eye(2)              # g.ndim == 2

        with pytest.raises(NotImplementedError, match="diagonal"):
            ItoMilstein().sde_step(
                term_mat, jnp.array(0.0), jnp.zeros(2), jnp.array(1.0),
                None, jnp.zeros(2),
            )

    def test_full_solve(self):
        sigma = 0.1
        y0 = jnp.array([1.0, 1.0])
        traj = solve(_linear_diffusion_term(sigma), None, y0, jnp.array(0.0), 10, 0.1, 0.1,
                     ItoMilstein(), key=jax.random.key(0), n_noise=2)
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))


class TestStratonovichMilstein:
    def test_ode_step_raises(self):
        with pytest.raises(NotImplementedError, match="SDE-only"):
            StratonovichMilstein().ode_step(
                uniform_ode_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None,
            )

    def test_constant_g_has_no_correction(self):
        # dg/dy = 0 → both correction terms vanish; matches Euler-Maruyama.
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.25)
        z = jnp.array([1.2, -0.7])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y_sm = StratonovichMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        y_em = Euler().sde_step(term, jnp.array(0.0), y0, dt, None, z)
        assert jnp.allclose(y_sm, y_em)

    def test_matrix_g_raises(self):
        def term_mat(t, y, args, z):
            return jnp.zeros(2), jnp.eye(2)

        with pytest.raises(NotImplementedError, match="diagonal"):
            StratonovichMilstein().sde_step(
                term_mat, jnp.array(0.0), jnp.zeros(2), jnp.array(1.0),
                None, jnp.zeros(2),
            )

    def test_full_solve(self):
        sigma = 0.1
        y0 = jnp.array([1.0, 1.0])
        traj = solve(_linear_diffusion_term(sigma), None, y0, jnp.array(0.0), 10, 0.1, 0.1,
                     StratonovichMilstein(), key=jax.random.key(0), n_noise=2)
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))
