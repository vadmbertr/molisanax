"""ODE and SDE solvers: Euler and Heun with a unified lax.scan integration loop.

Term API
--------
ODE term: ``f(t, y, args) -> Float[Array, "2"]``
    Returns velocity [dlat/dt, dlon/dt] in degrees/second.

SDE term: ``f(t, y, args) -> tuple[Float[Array, "2"], Float[Array, "2"]]``
    Returns (drift, noise_amplitude), both in degrees/second.
    The SDE step is: dy = (drift + noise_amplitude * z) * dt
    where z is a pre-sampled standard-normal vector passed from outside the
    scan loop. All noise is drawn before integration begins.

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
        """Advance SDE state y by one step using pre-sampled noise z ~ N(0, I_2)."""
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
        # Predictor using the pre-sampled noise z
        f0, g0 = term(t, y, args)
        y_pred = y + (f0 + g0 * z) * dt
        # Corrector — reuse the same z
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
        y_new = solver.ode_step(term, t, y, dt, args)
        return y_new, y_new

    _, ys = jax.lax.scan(body, y0, ts[:-1])
    return jnp.concatenate([y0[None], ys], axis=0)


def _run_sde(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
    noise: Float[Array, "samples steps 2"],
) -> Float[Array, "samples time 2"]:
    """Integrate an SDE ensemble. noise must be pre-sampled: shape (S, n_steps, 2)."""
    dt = ts[1] - ts[0]

    def solve_one(noise_i: Float[Array, "steps 2"]) -> Float[Array, "time 2"]:
        @jax.checkpoint
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, z = inputs
            y_new = solver.sde_step(term, t, y, dt, args, z)
            return y_new, y_new

        _, ys = jax.lax.scan(body, y0, (ts[:-1], noise_i))
        return jnp.concatenate([y0[None], ys], axis=0)

    return jax.vmap(solve_one)(noise)


def solve(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver | None = None,
    *,
    key: Key[Array, ""] | None = None,
    n_samples: int | None = None,
    noise: Float[Array, "..."] | None = None,
    adjoint: Literal["recursive_checkpoint", "forward"] = "recursive_checkpoint",
) -> Float[Array, "time 2"] | Float[Array, "samples time 2"]:
    """Integrate a trajectory from ts[0] to ts[-1] with constant step size.

    Automatically detects ODE vs SDE mode from the term's return type:

    - ODE: ``term(t, y, args) -> Float[Array, "2"]``
      Returns a single velocity. ``key``, ``n_samples``, and ``noise`` are ignored.

    - SDE: ``term(t, y, args) -> tuple[Float[Array, "2"], Float[Array, "2"]]``
      Returns ``(drift, noise_amplitude)``, both in deg/s.
      Step: ``dy = (drift + noise_amplitude * z) * dt``.
      All noise samples ``z`` are drawn **before** the integration loop begins.

    Noise for SDE mode can be supplied in two ways:

    1. **Pre-sampled** (recommended): pass ``noise`` directly.
       - Shape ``(n_steps, 2)`` → single realisation, returns ``(T, 2)``.
       - Shape ``(S, n_steps, 2)`` → ensemble, returns ``(S, T, 2)``.
       ``key`` is not needed when ``noise`` is provided.

    2. **Auto-sampled**: pass ``key`` (and optionally ``n_samples``).
       A single ``jax.random.normal`` call draws the full
       ``(n_samples, n_steps, 2)`` array before vmap and scan.

    Args:
        term: Dynamics callable. Return type determines ODE vs SDE mode.
        args: Arbitrary JAX pytree passed through to term (e.g. a Dataset).
        y0: Initial state [lat, lon] in degrees, shape (2,).
        ts: Equally spaced output times in seconds, shape (T,).
        solver: Solver instance. Defaults to Heun().
        key: PRNG key for auto-sampling noise. Required for SDE when noise=None.
        n_samples: Number of realisations for auto-sampled SDE. Defaults to 1.
        noise: Pre-sampled noise array of shape ``(n_steps, 2)`` (single) or
            ``(n_samples, n_steps, 2)`` (ensemble). When provided, ``key`` is
            not used. The noise values are used as-is (any distribution is valid).
        adjoint: AD strategy for ODE mode only. "recursive_checkpoint" uses
            lax.scan (discretise-then-optimise). "forward" is compatible with
            jax.jvp.

    Returns:
        - ODE: shape ``(T, 2)``.
        - SDE, single realisation: shape ``(T, 2)``.
        - SDE, ensemble: shape ``(S, T, 2)``.
    """
    if solver is None:
        solver = Heun()

    if not _is_sde_term(term, y0, args):
        return _run_ode(term, args, y0, ts, solver)

    # SDE path — normalise noise to (S, n_steps, 2) then run once
    n_steps = ts.shape[0] - 1

    if noise is not None:
        squeeze = noise.ndim == 2          # single realisation → remove S axis on output
        noise_3d = noise[None] if squeeze else noise
    elif key is not None:
        squeeze = n_samples is None
        n = 1 if squeeze else n_samples
        noise_3d = jr.normal(key, shape=(n, n_steps, 2), dtype=y0.dtype)
    else:
        raise ValueError(
            "SDE term detected (term returns a tuple) but neither 'noise' nor 'key' "
            "was provided. Pass noise=jnp.array(...) or key=jax.random.key(seed)."
        )

    ensemble = _run_sde(term, args, y0, ts, solver, noise_3d)
    return ensemble[0] if squeeze else ensemble
