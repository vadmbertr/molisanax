"""Along-trajectory metrics for evaluating Lagrangian simulation quality."""

import jax
import jax.numpy as jnp

from ._types import Array, Float
from .geo import haversine, safe_divide

__all__ = [
    "separation_distance",
    "normalized_separation_distance",
    "liu_index",
]


def _arc_lengths(traj: Float[Array, "time 2"]) -> Float[Array, " time"]:
    """Step-by-step great-circle arc lengths of a trajectory, in metres.

    Returns array of shape (T,) where element 0 is 0 (no step before the first point).
    """
    dists = jax.vmap(haversine)(traj[:-1], traj[1:])
    return jnp.concatenate([jnp.zeros((1,)), dists])


def separation_distance(
    y: Float[Array, "time 2"],
    y_ref: Float[Array, "time 2"],
) -> Float[Array, " time"]:
    """Point-wise great-circle distance between predicted and reference trajectories.

    Args:
        y: Predicted trajectory, shape (T, 2), [lat, lon] in degrees.
        y_ref: Reference trajectory, shape (T, 2), [lat, lon] in degrees.

    Returns:
        Distance at each time step in metres, shape (T,).
    """
    return jax.vmap(haversine)(y, y_ref)


def normalized_separation_distance(
    y: Float[Array, "time 2"],
    y_ref: Float[Array, "time 2"],
) -> Float[Array, " time"]:
    """Instantaneous separation distance normalised by cumulative reference arc length.

    Defined as: sep_dist(t) / cumsum(arc_length_ref)[t].

    Args:
        y: Predicted trajectory, shape (T, 2).
        y_ref: Reference trajectory, shape (T, 2).

    Returns:
        Normalised separation at each time step (dimensionless), shape (T,).
    """
    sep = separation_distance(y, y_ref)
    cum_ref_len = jnp.cumsum(_arc_lengths(y_ref))
    return safe_divide(sep, cum_ref_len)


def liu_index(
    y: Float[Array, "time 2"],
    y_ref: Float[Array, "time 2"],
) -> Float[Array, " time"]:
    """Liu Index: cumulative separation normalised by cumulative reference arc length.

    Defined as: cumsum(sep_dist)[t] / cumsum(arc_length_ref)[t].

    Reference: Liu & Weisberg (2011), J. Geophys. Res.

    Args:
        y: Predicted trajectory, shape (T, 2).
        y_ref: Reference trajectory, shape (T, 2).

    Returns:
        Liu Index at each time step (dimensionless), shape (T,).
    """
    sep = separation_distance(y, y_ref)
    cum_sep = jnp.cumsum(sep)
    cum_ref_len = jnp.cumsum(_arc_lengths(y_ref))
    return safe_divide(cum_sep, cum_ref_len)
