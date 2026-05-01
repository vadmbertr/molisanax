"""ODE and SDE solvers: Euler and Heun with a unified lax.scan integration loop.

Term API
--------
ODE term: ``f(t, y, args) -> Float[Array, "2"]``
    Returns velocity [dlat/dt, dlon/dt] in degrees/second.

SDE term: ``f(t, y, args) -> tuple[Float[Array, "2"], Float[Array, "2"]]``
    Returns (drift, noise_amplitude), both in degrees/second.
    The SDE step is: dy = (drift + noise_amplitude * z) * dt
    where z ~ N(0, I_2) is drawn fresh at each step by the solver.

The solver detects which mode to use by probing the term's output structure
with jax.eval_shape before entering the jit-compiled scan loop.
"""

from __future__ import annotations

import abc
from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ._types import Array, Float, Key, PyTree

__all__ = [
    "AbstractSolver",
    "Euler",
    "Heun",
    "solve",
]


class AbstractSolver(eqx.Module):
    """Abstract base for fixed-step ODE/SDE solvers."""

    @abc.abstractmethod
    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """Advance ODE state y by one step of size dt."""
        ...

    @abc.abstractmethod
    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        """Advance SDE state y by one step. z ~ N(0, I_2) is provided by the solver."""
        ...


class Euler(AbstractSolver):
    """Euler / Euler-Maruyama solver."""

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        return y + term(t, y, args) * dt

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        f, g = term(t, y, args)
        return y + (f + g * z) * dt


class Heun(AbstractSolver):
    """Heun (explicit second-order) solver."""

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        k1 = term(t, y, args)
        k2 = term(t + dt, y + k1 * dt, args)
        return y + 0.5 * (k1 + k2) * dt

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        # Predictor (Euler step with the same noise sample z)
        f0, g0 = term(t, y, args)
        y_pred = y + (f0 + g0 * z) * dt
        # Corrector
        f1, g1 = term(t + dt, y_pred, args)
        return y + 0.5 * ((f0 + f1) + (g0 + g1) * z) * dt


def _is_sde_term(term: Callable, y0: Float[Array, "2"], args: PyTree) -> bool:
    """Probe term output structure to detect SDE (tuple return) vs ODE (array return)."""
    dummy_t = jnp.zeros((), dtype=y0.dtype)
    dummy_y = jnp.zeros_like(y0)
    out = jax.eval_shape(term, dummy_t, dummy_y, args)
    return isinstance(out, tuple)


def _run_ode(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
) -> Float[Array, "time 2"]:
    dt = ts[1] - ts[0]

    @jax.checkpoint
    def body(y: Float[Array, "2"], t: Float[Array, ""]) -> tuple:
        return solver.ode_step(term, t, y, dt, args), solver.ode_step(term, t, y, dt, args)

    # Avoid computing the step twice: carry and output are the same value
    @jax.checkpoint
    def body_scan(y: Float[Array, "2"], t: Float[Array, ""]) -> tuple:
        y_new = solver.ode_step(term, t, y, dt, args)
        return y_new, y_new

    _, ys = jax.lax.scan(body_scan, y0, ts[:-1])
    return jnp.concatenate([y0[None], ys], axis=0)


def _run_sde(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
    key: Key[Array, ""],
    n_samples: int,
) -> Float[Array, "samples time 2"]:
    dt = ts[1] - ts[0]
    n_steps = ts.shape[0] - 1

    def solve_one(subkey: Key[Array, ""]) -> Float[Array, "time 2"]:
        # Pre-sample all noise increments z_k ~ N(0, I_2), one per step
        noise = jr.normal(subkey, shape=(n_steps, 2), dtype=y0.dtype)

        @jax.checkpoint
        def body_scan(
            y: Float[Array, "2"],
            inputs: tuple,
        ) -> tuple:
            t, z = inputs
            y_new = solver.sde_step(term, t, y, dt, args, z)
            return y_new, y_new

        _, ys = jax.lax.scan(body_scan, y0, (ts[:-1], noise))
        return jnp.concatenate([y0[None], ys], axis=0)

    keys = jr.split(key, n_samples)
    return jax.vmap(solve_one)(keys)


def solve(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver | None = None,
    *,
    key: Key[Array, ""] | None = None,
    n_samples: int | None = None,
    adjoint: Literal["recursive_checkpoint", "forward"] = "recursive_checkpoint",
) -> Float[Array, "time 2"] | Float[Array, "samples time 2"]:
    """Integrate a trajectory from ts[0] to ts[-1] with constant step size.

    Automatically detects ODE vs SDE mode from the term's return type:

    - ODE: ``term(t, y, args) -> Float[Array, "2"]``
      Returns a single velocity array. Use with ``jax.grad`` or ``jax.jvp``.

    - SDE: ``term(t, y, args) -> tuple[Float[Array, "2"], Float[Array, "2"]]``
      Returns ``(drift, noise_amplitude)``. The solver draws ``z ~ N(0, I_2)``
      at each step and computes ``dy = (drift + noise_amplitude * z) * dt``.
      Requires ``key`` to be provided.

    Args:
        term: Dynamics callable. Return type determines ODE vs SDE mode.
        args: Arbitrary JAX pytree passed through to term (e.g. a Dataset).
        y0: Initial state [lat, lon] in degrees, shape (2,).
        ts: Equally spaced output times in seconds, shape (T,).
        solver: Solver instance. Defaults to Heun().
        key: PRNG key. Required for SDE mode. Ignored for ODE.
        n_samples: Number of independent realisations (SDE only).
            If None and SDE mode is detected, defaults to 1 (single trajectory).
        adjoint: AD strategy for ODE mode. "recursive_checkpoint" differentiates
            through lax.scan (discretise-then-optimise). "forward" is compatible
            with jax.jvp. Ignored for SDE (use forward-mode for SDE gradients).

    Returns:
        - ODE: Float[Array, "time 2"] — shape (T, 2).
        - SDE with n_samples=None: Float[Array, "time 2"] — single realisation, shape (T, 2).
        - SDE with n_samples=N: Float[Array, "samples time 2"] — ensemble, shape (N, T, 2).
    """
    if solver is None:
        solver = Heun()

    if _is_sde_term(term, y0, args):
        if key is None:
            raise ValueError(
                "SDE term detected (term returns a tuple) but no PRNG key was provided. "
                "Pass key=jax.random.key(seed)."
            )
        n = n_samples if n_samples is not None else 1
        ensemble = _run_sde(term, args, y0, ts, solver, key, n)
        # Return (T, 2) when caller did not request an ensemble
        if n_samples is None:
            return ensemble[0]
        return ensemble
    else:
        return _run_ode(term, args, y0, ts, solver)
