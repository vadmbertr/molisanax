"""ODE and SDE solvers with a unified lax.scan integration loop.

Solver lineup
-------------
ODE solvers (both ``ode_step`` and ``sde_step``):

- :class:`Euler`, :class:`Heun`, :class:`RK4` — first / second / fourth-order
  explicit Runge–Kutta. ``sde_step`` interprets the term as a Stratonovich
  predictor–corrector that reuses the same Wiener increment across stages.

ODE-only solvers (raise on ``sde_step``):

- :class:`Tsit5` — Tsitouras 5(4)6 explicit RK, order 5 (fixed-step).
- :class:`Dopri5` — Dormand–Prince 5(4)7 explicit RK (FSAL), order 5
  (fixed-step).

SDE-only solvers (raise on ``ode_step``):

- :class:`EulerHeun` — diffrax-style Stratonovich predictor–corrector
  (diffusion-only predictor, Euler drift). Strong order 1.0.
- :class:`ItoMilstein`, :class:`StratonovichMilstein` — diagonal-noise Milstein
  schemes. Strong order 1.0. ``g`` must have shape ``(state_dim,)``; matrix
  diffusion raises.

Term API
--------
ODE term: ``f(t, y, args[, ctrl]) -> Float[Array, "2"]`` returns velocity
[dlat/dt, dlon/dt] in degrees/second.

SDE term: ``f(t, y, args[, ctrl]) -> tuple[Float[Array, "2"], Float[Array, "..."]]``
returns ``(drift, g)``. ``drift`` is the deterministic velocity and ``g`` is
the diffusion coefficient — the solver applies it as ``dy = drift*dt + g*dW``
with ``dW = sqrt(|dt|) * z`` and ``z ~ N(0, I_2)`` drawn internally. The term
never receives ``z``. Two ``g`` shapes are accepted:

- ``g.shape == (2,)`` — diagonal noise, noise step is ``g * dW`` componentwise.
- ``g.shape == (2, 2)`` — full 2×2 noise, noise step is ``g @ dW``.

The Milstein solvers require the diagonal form.

The optional ``ctrl`` argument is present when ``controls`` is passed to
:func:`solve`; the solver slices ``controls[i]`` at each step and forwards it
to the term. The term owns all interpretation and scaling of the slice.

Noise convention
----------------
Per-step Wiener increment is ``dW = sqrt(|dt|) * z`` with ``z ~ N(0, I_2)``
drawn internally; the SDE term never sees ``z``. Passing ``key`` to
:func:`solve` activates SDE mode. A single trajectory is produced by default;
pass ``n_samples > 1`` for an ensemble of independent realisations (returns
shape ``(n_samples, n_save+1, 2)`` via internal vmap over split keys).

Backwards-in-time integration is supported for all solvers: pass a negative
``int_dt`` (and matching negative ``save_dt``) to :func:`solve`. SDE backwards
integration is not a textbook construction, but remains finite because the
solver sign-abs-normalises the ``sqrt(dt)`` factor.
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
    "RK4",
    "Tsit5",
    "Dopri5",
    "EulerHeun",
    "ItoMilstein",
    "StratonovichMilstein",
    "solve",
]


def _apply_g(
    g: Float[Array, "..."],
    dW: Float[Array, "n_noise"],
) -> Float[Array, "2"]:
    """Apply a diffusion coefficient to a Wiener increment.

    Dispatches statically on ``g.ndim``: a 1-D ``g`` is a diagonal coefficient
    multiplied componentwise; a 2-D ``g`` is a full ``(2, n_noise)`` matrix
    contracted with ``dW``.
    """
    return g @ dW if g.ndim == 2 else g * dW


class AbstractSolver(eqx.Module):
    """Abstract base class for fixed-step ODE/SDE solvers.

    Subclasses implement :meth:`ode_step` (deterministic) and :meth:`sde_step`
    (stochastic, with a pre-sampled ``z``). Solvers that are specific to one
    mode raise :class:`NotImplementedError` from the other.
    """

    @abc.abstractmethod
    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """Advance the ODE state by one step of size ``dt``.

        Args:
            term: Drift callable ``f(t, y, args) -> Float[Array, "2"]`` returning
                velocity in degrees per second.
            t: Current time, in seconds.
            y: Current state ``[lat, lon]`` in degrees.
            dt: Step size in seconds.
            args: Arbitrary pytree forwarded to ``term``.

        Returns:
            Updated state ``[lat, lon]`` in degrees after one step.
        """
        ...

    @abc.abstractmethod
    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """Advance the SDE state by one step using a pre-sampled ``z``.

        Args:
            term: Stochastic dynamics callable ``f(t, y, args) -> (drift, g)``.
                ``drift`` is the deterministic velocity in degrees per second;
                ``g`` is the diffusion coefficient with shape ``(2,)``
                (diagonal) or ``(2, 2)`` (full matrix). The term never sees
                ``z``; the Wiener increment is applied by the solver.
            t: Current time, in seconds.
            y: Current state ``[lat, lon]`` in degrees.
            dt: Step size in seconds.
            args: Arbitrary pytree forwarded to ``term``.
            z: Standard-normal noise sample of shape ``(n_noise,)``. The Wiener
                increment used by the solver is ``dW = sqrt(|dt|) * z``.

        Returns:
            Updated state ``[lat, lon]`` in degrees after one step.
        """
        ...


class Euler(AbstractSolver):
    """Explicit Euler / Euler–Maruyama solver (first-order, fixed-step)."""

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """One Euler step: ``y_new = y + term(t, y, args) * dt``."""
        return y + term(t, y, args) * dt

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One Euler–Maruyama step: ``y + drift*dt + g*dW``."""
        f, g = term(t, y, args)
        dW = jnp.sqrt(jnp.abs(dt)) * z
        return y + f * dt + _apply_g(g, dW)


class Heun(AbstractSolver):
    """Heun (explicit second-order, two-stage Runge–Kutta) solver.

    Convergence order 2 in the ODE case. The SDE step is a Stratonovich
    predictor–corrector that reuses the same ``dW`` in both stages.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """One Heun (trapezoidal) step."""
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
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One Stratonovich Heun step (same ``dW`` in predictor and corrector)."""
        f0, g0 = term(t, y, args)
        dW = jnp.sqrt(jnp.abs(dt)) * z
        v0 = f0 * dt + _apply_g(g0, dW)
        f1, g1 = term(t + dt, y + v0, args)
        v1 = f1 * dt + _apply_g(g1, dW)
        return y + 0.5 * (v0 + v1)


class RK4(AbstractSolver):
    """Classical fourth-order Runge–Kutta solver (four stages, fixed-step).

    Convergence order 4 in the ODE case. The SDE step reuses the same ``dW``
    across all four stages, yielding a Stratonovich-consistent scheme whose
    strong order is limited by the noise structure.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """One classical RK4 step."""
        half = dt * 0.5
        k1 = term(t, y, args)
        k2 = term(t + half, y + k1 * half, args)
        k3 = term(t + half, y + k2 * half, args)
        k4 = term(t + dt,   y + k3 * dt,   args)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One stochastic RK4 step (Stratonovich, single ``dW`` across stages)."""
        half = dt * 0.5
        dW = jnp.sqrt(jnp.abs(dt)) * z

        def velocity(t_, y_):
            f_, g_ = term(t_, y_, args)
            return f_ * dt + _apply_g(g_, dW)

        v1 = velocity(t,        y)
        v2 = velocity(t + half, y + v1 * 0.5)
        v3 = velocity(t + half, y + v2 * 0.5)
        v4 = velocity(t + dt,   y + v3)
        return y + (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0


# --- Tsit5 coefficients (Tsitouras 2011, RK5(4)6) ------------------------------
# Source: Ch. Tsitouras, "Runge–Kutta pairs of order 5(4) satisfying only the
# first column simplifying assumption", Comput. Math. Appl. 62 (2011), 770-775.
# Same values used by DifferentialEquations.jl and diffrax.
_TSIT5_A21 = 0.161
_TSIT5_A31 = -0.008480655492356989
_TSIT5_A32 = 0.335480655492357
_TSIT5_A41 = 2.8971530571054935
_TSIT5_A42 = -6.359448489975075
_TSIT5_A43 = 4.3622954328695815
_TSIT5_A51 = 5.325864828439257
_TSIT5_A52 = -11.748883564062828
_TSIT5_A53 = 7.4955393428898365
_TSIT5_A54 = -0.09249506636175525
_TSIT5_A61 = 5.86145544294642
_TSIT5_A62 = -12.92096931784711
_TSIT5_A63 = 8.159367898576159
_TSIT5_A64 = -0.071584973281401
_TSIT5_A65 = -0.028269050394068383

_TSIT5_C2 = 0.161
_TSIT5_C3 = 0.327
_TSIT5_C4 = 0.9
_TSIT5_C5 = 0.9800255409045097
_TSIT5_C6 = 1.0

_TSIT5_B1 = 0.09646076681806523
_TSIT5_B2 = 0.01
_TSIT5_B3 = 0.4798896504144996
_TSIT5_B4 = 1.379008574103742
_TSIT5_B5 = -3.290069515436081
_TSIT5_B6 = 2.324710524099774


class Tsit5(AbstractSolver):
    """Tsitouras 5(4)6 explicit Runge–Kutta (ODE-only, fixed-step, order 5).

    Six stages, no embedded error estimator (the 4th-order companion row of
    Tsitouras 2011 is unused since we are fixed-step).
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """One Tsit5 step (5th-order weights only)."""
        k1 = term(t, y, args)
        k2 = term(t + _TSIT5_C2 * dt, y + dt * (_TSIT5_A21 * k1), args)
        k3 = term(t + _TSIT5_C3 * dt, y + dt * (_TSIT5_A31 * k1 + _TSIT5_A32 * k2), args)
        k4 = term(t + _TSIT5_C4 * dt, y + dt * (_TSIT5_A41 * k1 + _TSIT5_A42 * k2 + _TSIT5_A43 * k3), args)
        k5 = term(t + _TSIT5_C5 * dt,
                  y + dt * (_TSIT5_A51 * k1 + _TSIT5_A52 * k2 + _TSIT5_A53 * k3 + _TSIT5_A54 * k4),
                  args)
        k6 = term(t + _TSIT5_C6 * dt,
                  y + dt * (_TSIT5_A61 * k1 + _TSIT5_A62 * k2 + _TSIT5_A63 * k3 + _TSIT5_A64 * k4 + _TSIT5_A65 * k5),
                  args)
        return y + dt * (
            _TSIT5_B1 * k1 + _TSIT5_B2 * k2 + _TSIT5_B3 * k3
            + _TSIT5_B4 * k4 + _TSIT5_B5 * k5 + _TSIT5_B6 * k6
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "Tsit5 is an ODE-only solver; use Euler, Heun, RK4, EulerHeun, "
            "ItoMilstein, or StratonovichMilstein for SDEs."
        )


class Dopri5(AbstractSolver):
    """Dormand–Prince 5(4)7 explicit Runge–Kutta (ODE-only, fixed-step, order 5).

    Seven stages with the first-same-as-last property; here we use the
    5th-order row only.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        """One Dopri5 step (5th-order weights only)."""
        k1 = term(t, y, args)
        k2 = term(t + dt * (1.0 / 5.0), y + dt * (1.0 / 5.0) * k1, args)
        k3 = term(t + dt * (3.0 / 10.0),
                  y + dt * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2),
                  args)
        k4 = term(t + dt * (4.0 / 5.0),
                  y + dt * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3),
                  args)
        k5 = term(t + dt * (8.0 / 9.0),
                  y + dt * (
                      19372.0 / 6561.0 * k1 - 25360.0 / 2187.0 * k2
                      + 64448.0 / 6561.0 * k3 - 212.0 / 729.0 * k4
                  ),
                  args)
        k6 = term(t + dt,
                  y + dt * (
                      9017.0 / 3168.0 * k1 - 355.0 / 33.0 * k2
                      + 46732.0 / 5247.0 * k3 + 49.0 / 176.0 * k4
                      - 5103.0 / 18656.0 * k5
                  ),
                  args)
        return y + dt * (
            35.0 / 384.0 * k1
            + 500.0 / 1113.0 * k3
            + 125.0 / 192.0 * k4
            - 2187.0 / 6784.0 * k5
            + 11.0 / 84.0 * k6
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "Dopri5 is an ODE-only solver; use Euler, Heun, RK4, EulerHeun, "
            "ItoMilstein, or StratonovichMilstein for SDEs."
        )


class EulerHeun(AbstractSolver):
    """Stochastic Euler–Heun solver (SDE-only, Stratonovich, strong order 1.0).

    Matches diffrax's ``EulerHeun`` algorithm: the predictor uses *diffusion
    only* and the drift is applied once Euler-style. Accepts both diagonal
    (``g.shape == (2,)``) and full (``g.shape == (2, 2)``) diffusion shapes.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "EulerHeun is an SDE-only solver; use Euler, Heun, RK4, Tsit5, "
            "or Dopri5 for ODEs."
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One stochastic Euler–Heun step."""
        f0, g0 = term(t, y, args)
        dW = jnp.sqrt(jnp.abs(dt)) * z
        diff0 = _apply_g(g0, dW)
        y_pred = y + diff0
        _, g1 = term(t + dt, y_pred, args)
        diff1 = _apply_g(g1, dW)
        return y + f0 * dt + 0.5 * (diff0 + diff1)


def _milstein_correction(
    term: Callable,
    t: Float[Array, ""],
    y: Float[Array, "2"],
    args: PyTree,
    g: Float[Array, "2"],
    dW: Float[Array, "n_noise"],
) -> Float[Array, "2"]:
    """Diagonal-noise Milstein cross-term ``0.5 * g * (∂g/∂y_i) * dW^2``.

    Returns the ``0.5 * g * (∂g_i/∂y_i) * dW**2`` vector (Stratonovich form).
    Itô subtracts ``0.5 * g * (∂g_i/∂y_i) * dt`` on top.
    """
    def g_fn(y_):
        _, g_out = term(t, y_, args)
        return g_out

    dgdy = jax.jacfwd(g_fn)(y)            # (2, 2)
    dgdy_diag = jnp.diag(dgdy)
    return 0.5 * g * dgdy_diag * dW ** 2


class ItoMilstein(AbstractSolver):
    """Itô Milstein solver (SDE-only, diagonal noise, strong order 1.0).

    Requires ``g.shape == (2,)``. Raises :class:`NotImplementedError` for
    matrix-valued ``g`` — use :class:`EulerHeun` for general noise.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "ItoMilstein is an SDE-only solver; use Euler, Heun, RK4, Tsit5, "
            "or Dopri5 for ODEs."
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One Itô Milstein step: ``y + f*dt + g*dW + 0.5*g*(∂g/∂y)*(dW**2 - dt)``."""
        f, g = term(t, y, args)
        if g.ndim != 1:
            raise NotImplementedError(
                "ItoMilstein requires diagonal noise (g.shape == (2,)); "
                f"got g.shape == {g.shape}. Use EulerHeun for matrix diffusion."
            )
        dW = jnp.sqrt(jnp.abs(dt)) * z
        cross = _milstein_correction(term, t, y, args, g, dW)
        def g_fn(y_):
            _, g_out = term(t, y_, args)
            return g_out
        dgdy_diag = jnp.diag(jax.jacfwd(g_fn)(y))
        ito_drift = -0.5 * g * dgdy_diag * dt
        return y + f * dt + g * dW + cross + ito_drift


class StratonovichMilstein(AbstractSolver):
    """Stratonovich Milstein solver (SDE-only, diagonal noise, strong order 1.0).

    Requires ``g.shape == (2,)``. Differs from :class:`ItoMilstein` by the
    absence of the ``-0.5 * g * (∂g/∂y) * dt`` Itô-to-Stratonovich correction.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "StratonovichMilstein is an SDE-only solver; use Euler, Heun, RK4, "
            "Tsit5, or Dopri5 for ODEs."
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One Stratonovich Milstein step: ``y + f*dt + g*dW + 0.5*g*(∂g/∂y)*dW**2``."""
        f, g = term(t, y, args)
        if g.ndim != 1:
            raise NotImplementedError(
                "StratonovichMilstein requires diagonal noise (g.shape == (2,)); "
                f"got g.shape == {g.shape}. Use EulerHeun for matrix diffusion."
            )
        dW = jnp.sqrt(jnp.abs(dt)) * z
        cross = _milstein_correction(term, t, y, args, g, dW)
        return y + f * dt + g * dW + cross


def _run_ode(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
    controls: PyTree | None = None,
) -> Float[Array, "time 2"]:
    dt = ts[1] - ts[0]

    if controls is None:
        @jax.checkpoint
        def body(y: Float[Array, "2"], t: Float[Array, ""]) -> tuple:
            y_new = solver.ode_step(term, t, y, dt, args)
            return y_new, y_new

        _, ys = jax.lax.scan(body, y0, ts[:-1])
    else:
        @jax.checkpoint
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, ctrl = inputs
            bound = lambda t_, y_, a_: term(t_, y_, a_, ctrl)
            y_new = solver.ode_step(bound, t, y, dt, args)
            return y_new, y_new

        _, ys = jax.lax.scan(body, y0, (ts[:-1], controls))

    return jnp.concatenate([y0[None], ys], axis=0)


def _run_sde(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
    z_seq: Float[Array, "steps 2"],
    controls: PyTree | None = None,
) -> Float[Array, "time 2"]:
    dt = ts[1] - ts[0]

    if controls is None:
        @jax.checkpoint
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, z = inputs
            y_new = solver.sde_step(term, t, y, dt, args, z)
            return y_new, y_new

        _, ys = jax.lax.scan(body, y0, (ts[:-1], z_seq))
    else:
        @jax.checkpoint
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, z, ctrl = inputs
            bound = lambda t_, y_, a_: term(t_, y_, a_, ctrl)
            y_new = solver.sde_step(bound, t, y, dt, args, z)
            return y_new, y_new

        _, ys = jax.lax.scan(body, y0, (ts[:-1], z_seq, controls))

    return jnp.concatenate([y0[None], ys], axis=0)


def solve(
    term: Callable,
    args: PyTree,
    y0: Float[Array, "2"],
    t0: Float[Array, ""],
    n_save: int,
    int_dt: float,
    save_dt: float,
    solver: AbstractSolver | None = None,
    *,
    controls: PyTree | None = None,
    key: Key[Array, ""] | None = None,
    n_samples: int = 1,
    adjoint: Literal["recursive_checkpoint", "forward"] = "recursive_checkpoint",
) -> Array:
    """Integrate a trajectory for ``n_save`` output intervals starting at ``t0``.

    ODE mode (default, no ``key``): ``term(t, y, args[, ctrl])`` returns velocity.
    SDE mode (pass ``key``): ``term(t, y, args[, ctrl])`` returns ``(drift, g)``;
    the solver draws ``z ~ N(0, I_2)`` and applies ``dW = sqrt(|int_dt|) * z``
    internally. The optional ``ctrl`` argument is present when ``controls`` is
    provided — the solver slices it at each step; the term owns its interpretation.

    The solver runs on a fine integration grid of ``n_fine = n_save * n_substeps``
    steps (where ``n_substeps = round(save_dt / int_dt)``), then slices every
    ``n_substeps`` steps to produce the ``n_save + 1`` saved states.

    **Ensemble**: pass ``n_samples > 1`` in SDE mode; the key is split internally.
    **Perturbed ODE**: use ODE+controls and
    ``jax.vmap(lambda c: solve(..., controls=c))(controls_batch)``.

    Args:
        term: Dynamics callable ``f(t, y, args[, ctrl])``. ODE: returns velocity.
            SDE: returns ``(drift, g)`` where ``g`` is the diffusion coefficient,
            shape ``(2,)`` diagonal or ``(2, 2)`` full matrix.
        args: Arbitrary JAX pytree passed through to term (e.g. a Dataset).
        y0: Initial state [lat, lon] in degrees, shape (2,).
        t0: Start time in seconds. JAX scalar — can change between calls without
            recompilation. The implicit end time is ``t0 + n_save * save_dt``.
        n_save: Number of output intervals (static). Output has shape
            ``(n_save + 1, 2)`` including the initial state.
        int_dt: Integration step size in seconds (static). Use a negative value
            for backward-in-time integration.
        save_dt: Output interval in seconds (static). Must be an integer multiple
            of ``int_dt`` (same sign). ``n_substeps = round(save_dt / int_dt) >= 1``.
        solver: Solver instance. Defaults to Heun().
        controls: Per-step pytree with leading axis ``n_fine``. Sliced at each
            integration step and forwarded as the 4th argument to the term, in
            both ODE and SDE modes.
        key: PRNG key for SDE mode. When provided, draws ``z ~ N(0, I_2)`` of
            shape ``(n_fine, 2)`` and runs in SDE mode.
        n_samples: Number of independent SDE realisations (default 1). Ignored in
            ODE mode. When > 1, the key is split and trajectories are vmapped;
            output shape is ``(n_samples, n_save + 1, 2)``.
        adjoint: AD strategy for ODE mode only. "recursive_checkpoint" uses
            lax.scan (discretise-then-optimise). "forward" is compatible with
            jax.jvp.

    Returns:
        Shape ``(n_save + 1, 2)`` in ODE mode or SDE with ``n_samples == 1``.
        Shape ``(n_samples, n_save + 1, 2)`` in SDE mode with ``n_samples > 1``.
    """
    if solver is None:
        solver = Heun()

    n_substeps = round(save_dt / int_dt)
    if n_substeps < 1:
        raise ValueError(
            f"save_dt/int_dt must be >= 1 (got {n_substeps}). "
            "For backward integration both int_dt and save_dt must be negative."
        )
    if abs(n_substeps * int_dt - save_dt) > 1e-8 * abs(save_dt):
        raise ValueError(
            f"save_dt ({save_dt}) must be an integer multiple of int_dt ({int_dt})."
        )
    n_fine = n_save * n_substeps
    ts_fine = t0 + jnp.arange(n_fine + 1) * int_dt

    if key is not None:
        if n_samples == 1:
            z = jr.normal(key, shape=(n_fine, 2), dtype=y0.dtype)
            result = _run_sde(term, args, y0, ts_fine, solver, z, controls)
            return result[::n_substeps]
        else:
            keys = jr.split(key, n_samples)
            def _single(k):
                z = jr.normal(k, shape=(n_fine, 2), dtype=y0.dtype)
                return _run_sde(term, args, y0, ts_fine, solver, z, controls)[::n_substeps]
            return jax.vmap(_single)(keys)
    else:
        result = _run_ode(term, args, y0, ts_fine, solver, controls)
        return result[::n_substeps]
