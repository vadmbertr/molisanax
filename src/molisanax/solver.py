"""ODE and SDE solvers: Euler and Heun with lax.scan integration loop."""

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
    "solve_ode",
    "solve_sde",
]


class AbstractSolver(eqx.Module):
    """Abstract base for fixed-step ODE solvers."""

    @abc.abstractmethod
    def step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """Advance state y by one step of size dt using the given term."""
        ...

    @abc.abstractmethod
    def sde_step(
        self,
        drift: Callable,
        diffusion: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        dw: Float[Array, " n"],
    ) -> Float[Array, "2"]:
        """Advance SDE state y by one step. dw is the noise increment ~ N(0, sqrt(dt))."""
        ...


class Euler(AbstractSolver):
    """Euler (explicit first-order) solver for ODE and SDE (Euler-Maruyama for SDE)."""

    def step(
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
        drift: Callable,
        diffusion: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        dw: Float[Array, " n"],
    ) -> Float[Array, "2"]:
        det = drift(t, y, args) * dt
        stoch = diffusion(t, y, args) @ dw
        return y + det + stoch


class Heun(AbstractSolver):
    """Heun (explicit second-order) solver. For SDEs uses a Stratonovich-consistent predictor-corrector."""

    def step(
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
        drift: Callable,
        diffusion: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        dw: Float[Array, " n"],
    ) -> Float[Array, "2"]:
        g0 = diffusion(t, y, args)
        f0 = drift(t, y, args)
        y_pred = y + f0 * dt + g0 @ dw
        f1 = drift(t + dt, y_pred, args)
        g1 = diffusion(t + dt, y_pred, args)
        det = 0.5 * (f0 + f1) * dt
        stoch = 0.5 * (g0 + g1) @ dw
        return y + det + stoch


def solve_ode(
    term: Callable[[Float[Array, ""], Float[Array, "2"], PyTree], Float[Array, "2"]],
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver | None = None,
    *,
    adjoint: Literal["recursive_checkpoint", "forward"] = "recursive_checkpoint",
) -> Float[Array, "time 2"]:
    """Integrate an ODE from ts[0] to ts[-1] with constant step size.

    Args:
        term: Dynamics callable ``f(t, y, args) -> dy/dt`` in degrees/second.
        args: Arbitrary JAX pytree passed through to term unchanged.
        y0: Initial state [lat, lon] in degrees, shape (2,).
        ts: Equally spaced output times in seconds, shape (T,).
        solver: Solver instance (default: Heun()).
        adjoint: AD strategy. "recursive_checkpoint" differentiates through lax.scan
            (discretise-then-optimise). "forward" is compatible with jax.jvp.

    Returns:
        Trajectory array of shape (T, 2) where output[0] == y0.
    """
    if solver is None:
        solver = Heun()

    dt = ts[1] - ts[0]

    @jax.checkpoint
    def body(y: Float[Array, "2"], t: Float[Array, ""]) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        y_new = solver.step(term, t, y, dt, args)
        return y_new, y_new

    _, ys = jax.lax.scan(body, y0, ts[:-1])
    return jnp.concatenate([y0[None], ys], axis=0)


def solve_sde(
    drift: Callable[[Float[Array, ""], Float[Array, "2"], PyTree], Float[Array, "2"]],
    diffusion: Callable[[Float[Array, ""], Float[Array, "2"], PyTree], Float[Array, "2 n"]],
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    key: Key[Array, ""],
    n_samples: int,
    n_noise: int,
    solver: AbstractSolver | None = None,
) -> Float[Array, "samples time 2"]:
    """Integrate an SDE ensemble from ts[0] to ts[-1] with constant step size.

    Args:
        drift: Deterministic term ``f(t, y, args) -> dy/dt`` in degrees/second.
        diffusion: Stochastic term ``g(t, y, args) -> matrix of shape (2, n_noise)``.
        args: JAX pytree passed through to drift and diffusion.
        y0: Initial state [lat, lon] in degrees, shape (2,).
        ts: Equally spaced output times in seconds, shape (T,).
        key: PRNG key for noise generation.
        n_samples: Number of independent realisations.
        n_noise: Dimension of the noise process (columns of diffusion matrix).
        solver: Solver instance (default: Heun()).

    Returns:
        Ensemble trajectories of shape (n_samples, T, 2).
    """
    if solver is None:
        solver = Heun()

    dt = ts[1] - ts[0]
    n_steps = ts.shape[0] - 1

    def solve_one(subkey: Key[Array, ""]) -> Float[Array, "time 2"]:
        # Pre-sample all noise increments: dW_k ~ N(0, I) * sqrt(dt)
        noise = jr.normal(subkey, shape=(n_steps, n_noise)) * jnp.sqrt(dt)

        @jax.checkpoint
        def body(
            y: Float[Array, "2"],
            inputs: tuple[Float[Array, ""], Float[Array, " n"]],
        ) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
            t, dw = inputs
            y_new = solver.sde_step(drift, diffusion, t, y, dt, args, dw)
            return y_new, y_new

        _, ys = jax.lax.scan(body, y0, (ts[:-1], noise))
        return jnp.concatenate([y0[None], ys], axis=0)

    keys = jr.split(key, n_samples)
    return jax.vmap(solve_one)(keys)
