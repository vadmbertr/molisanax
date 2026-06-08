"""Along-trajectory metrics for evaluating Lagrangian simulation quality."""

import jax
import jax.numpy as jnp

from ._safe_math import safe_divide
from ._types import Array, Float
from .geo import haversine

__all__ = [
    "separation_distance",
    "normalized_separation_distance",
    "liu_index",
]


def _arc_lengths(traj: Float[Array, "time 2"]) -> Float[Array, " time"]:
    """Step-by-step great-circle arc lengths of a trajectory in metres.

    Returns shape (T,); element 0 is always 0 (no step before the first point).
    """
    dists = jax.vmap(haversine)(traj[:-1], traj[1:])
    return jnp.concatenate([jnp.zeros((1,), dtype=dists.dtype), dists])


def separation_distance(
    y: Float[Array, "* time 2"],
    y_ref: Float[Array, "time 2"],
    *,
    ensemble: bool = False,
) -> Float[Array, "* time"]:
    """Point-wise great-circle distance between predicted and reference trajectories.

    Args:
        y: Predicted trajectory, shape (T, 2). If ensemble=True, shape (S, T, 2).
        y_ref: Reference trajectory, shape (T, 2).
        ensemble: If True, y is treated as an ensemble of shape (S, T, 2) and the
            metric is computed independently for each member via vmap.

    Returns:
        Distance at each time step in metres. Shape (T,) or (S, T).
    """
    if ensemble:
        return jax.vmap(lambda yi: separation_distance(yi, y_ref, ensemble=False))(y)
    return jax.vmap(haversine)(y, y_ref)


def normalized_separation_distance(
    y: Float[Array, "* time 2"],
    y_ref: Float[Array, "time 2"],
    *,
    ensemble: bool = False,
) -> Float[Array, "* time"]:
    r"""Instantaneous separation distance normalised by cumulative reference arc length.

    .. math::

        \mathrm{NSD}(t) = \frac{\mathrm{sep\_dist}(t)}
        {\operatorname{cumsum}(\mathrm{arc\_length\_ref})[t]}

    Args:
        y: Predicted trajectory, shape (T, 2). If ensemble=True, shape (S, T, 2).
        y_ref: Reference trajectory, shape (T, 2).
        ensemble: If True, vmaps over the first (sample) axis.

    Returns:
        Normalised separation (dimensionless). Shape (T,) or (S, T).
    """
    if ensemble:
        return jax.vmap(
            lambda yi: normalized_separation_distance(yi, y_ref, ensemble=False)
        )(y)
    sep = separation_distance(y, y_ref)
    cum_ref_len = jnp.cumsum(_arc_lengths(y_ref))
    return safe_divide(sep, cum_ref_len)


def liu_index(
    y: Float[Array, "* time 2"],
    y_ref: Float[Array, "time 2"],
    *,
    ensemble: bool = False,
) -> Float[Array, "* time"]:
    r"""Liu Index: cumulative separation normalised by cumulative reference arc length.

    .. math::

        \mathrm{Liu}(t) = \frac{\operatorname{cumsum}(\mathrm{sep\_dist})[t]}
        {\operatorname{cumsum}(\mathrm{arc\_length\_ref})[t]}

    Reference: Liu & Weisberg (2011), J. Geophys. Res.

    Args:
        y: Predicted trajectory, shape (T, 2). If ensemble=True, shape (S, T, 2).
        y_ref: Reference trajectory, shape (T, 2).
        ensemble: If True, vmaps over the first (sample) axis.

    Returns:
        Liu Index (dimensionless). Shape (T,) or (S, T).
    """
    if ensemble:
        return jax.vmap(lambda yi: liu_index(yi, y_ref, ensemble=False))(y)
    sep = separation_distance(y, y_ref)
    cum_sep = jnp.cumsum(sep)
    cum_ref_len = jnp.cumsum(_arc_lengths(y_ref))
    return safe_divide(cum_sep, cum_ref_len)
