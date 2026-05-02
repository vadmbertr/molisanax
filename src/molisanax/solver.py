"""ODE and SDE solvers: Euler and Heun with a unified lax.scan integration loop.

Term API
--------
ODE term: ``f(t, y, args) -> Float[Array, "2"]``
    Returns velocity [dlat/dt, dlon/dt] in degrees/second.

SDE term: ``f(t, y, args, z) -> Float[Array, "2"]``
    Receives the pre-sampled noise vector z and returns the full velocity.
    The step is: dy = term(t, y, args, z) * dt.
    All noise is drawn before integration begins.

Mode detection is based on the call site: passing any of ``key``, ``noise``,
or ``n_noise`` to ``solve()`` selects SDE mode.  If none are passed, ODE mode
is assumed and the term is called without a noise argument.
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
        z: Float[Array, " n_noise"],
    ) -> Float[Array, "2"]:
        """Advance SDE state y by one step using pre-sampled noise z."""
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
        z: Float[Array, " n_noise"],
    ) -> Float[Array, "2"]:
        return y + term(t, y, args, z) * dt


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
        z: Float[Array, " n_noise"],
    ) -> Float[Array, "2"]:
        # Same z for predictor and corrector (Stratonovich-consistent).
        k1 = term(t, y, args, z)
        k2 = term(t + dt, y + k1 * dt, args, z)
        return y + 0.5 * (k1 + k2) * dt


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
    noise: Float[Array, "samples steps n_noise"],
) -> Float[Array, "samples time 2"]:
    """Integrate an SDE ensemble. noise must be pre-sampled: shape (S, n_steps, n_noise)."""
    dt = ts[1] - ts[0]

    def solve_one(noise_i: Float[Array, "steps n_noise"]) -> Float[Array, "time 2"]:
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
    n_noise: int | None = None,
    noise: Float[Array, "..."] | None = None,
    adjoint: Literal["recursive_checkpoint", "forward"] = "recursive_checkpoint",
) -> Float[Array, "time 2"] | Float[Array, "samples time 2"]:
    """Integrate a trajectory from ts[0] to ts[-1] with constant step size.

    Mode is selected by the caller:

    - **ODE** (default): none of ``key``, ``noise``, or ``n_noise`` are passed.
      ``term(t, y, args) -> Float[Array, "2"]`` returns velocity.

    - **SDE**: at least one of ``key``, ``noise``, or ``n_noise`` is passed.
      ``term(t, y, args, z) -> Float[Array, "2"]`` receives the pre-sampled
      noise vector ``z`` of shape ``(n_noise,)`` and returns the full velocity.
      The step is ``dy = term(t, y, args, z) * dt``.

    Noise for SDE mode can be supplied in two ways:

    1. **Pre-sampled** (recommended): pass ``noise`` directly.
       - Shape ``(n_steps, n_noise)`` → single realisation, returns ``(T, 2)``.
       - Shape ``(S, n_steps, n_noise)`` → ensemble, returns ``(S, T, 2)``.
       ``key`` and ``n_noise`` are not needed when ``noise`` is provided.

    2. **Auto-sampled**: pass ``key`` and ``n_noise`` (and optionally ``n_samples``).
       A single ``jax.random.normal`` call draws the full
       ``(n_samples, n_steps, n_noise)`` array before vmap and scan.

    Args:
        term: Dynamics callable. ODE: ``f(t, y, args)``. SDE: ``f(t, y, args, z)``.
        args: Arbitrary JAX pytree passed through to term (e.g. a Dataset).
        y0: Initial state [lat, lon] in degrees, shape (2,).
        ts: Equally spaced output times in seconds, shape (T,).
        solver: Solver instance. Defaults to Heun().
        key: PRNG key for auto-sampling noise (SDE only). Required when ``noise``
            is None in SDE mode.
        n_samples: Number of realisations for auto-sampled SDE. Defaults to 1.
        n_noise: Dimension of the noise vector ``z``. Required for auto-sampling;
            inferred from ``noise.shape[-1]`` when pre-sampled noise is provided.
        noise: Pre-sampled noise array of shape ``(n_steps, n_noise)`` (single) or
            ``(S, n_steps, n_noise)`` (ensemble). When provided, ``key`` and
            ``n_noise`` are not used.
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

    # ODE mode: none of the SDE-specific arguments are provided.
    if key is None and noise is None and n_noise is None:
        return _run_ode(term, args, y0, ts, solver)

    # SDE path — normalise noise to (S, n_steps, n_noise) then run once.
    n_steps = ts.shape[0] - 1

    if noise is not None:
        squeeze = noise.ndim == 2          # single realisation → remove S axis on output
        noise_3d = noise[None] if squeeze else noise
    elif key is not None:
        if n_noise is None:
            raise ValueError(
                "SDE mode: 'n_noise' is required when auto-sampling noise via 'key'. "
                "Pass n_noise=<latent dimension> or provide pre-sampled 'noise' instead."
            )
        squeeze = n_samples is None
        n = 1 if squeeze else n_samples
        noise_3d = jr.normal(key, shape=(n, n_steps, n_noise), dtype=y0.dtype)
    else:
        raise ValueError(
            "SDE mode (n_noise was provided) but neither 'key' nor 'noise' was given. "
            "Pass key=jax.random.key(seed) or noise=jnp.array(...)."
        )

    ensemble = _run_sde(term, args, y0, ts, solver, noise_3d)
    return ensemble[0] if squeeze else ensemble
