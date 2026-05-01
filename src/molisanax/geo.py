"""Geographic utilities: unit conversions, safe differentiable math, Haversine distance."""

import jax.numpy as jnp

from ._types import Array, Float

__all__ = [
    "EARTH_RADIUS",
    "safe_sqrt",
    "safe_log",
    "safe_divide",
    "haversine",
    "meters_to_degrees",
    "degrees_to_meters",
]

EARTH_RADIUS: float = 6_371_008.8  # metres


def safe_sqrt(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gradient-safe sqrt: returns 0 where x <= 0, avoids NaN gradients at 0."""
    mask = x > 0.0
    return jnp.where(mask, jnp.sqrt(jnp.where(mask, x, 1.0)), 0.0)


def safe_log(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gradient-safe log: returns -inf where x <= 0, avoids NaN gradients at 0."""
    mask = x > 0.0
    return jnp.where(mask, jnp.log(jnp.where(mask, x, 1.0)), -jnp.inf)


def safe_divide(
    a: Float[Array, "..."], b: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Gradient-safe divide: returns 0 where b == 0."""
    mask = b != 0.0
    return jnp.where(mask, a / jnp.where(mask, b, 1.0), 0.0)


def haversine(
    y1: Float[Array, "2"],
    y2: Float[Array, "2"],
) -> Float[Array, ""]:
    """Great-circle distance between two [lat, lon] points in degrees. Returns metres."""
    lat1 = jnp.radians(y1[0])
    lat2 = jnp.radians(y2[0])
    d = jnp.radians(y1 - y2)
    a = jnp.sin(d[0] / 2) ** 2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(d[1] / 2) ** 2
    c = 2.0 * jnp.arctan2(safe_sqrt(a), safe_sqrt(1.0 - a))
    return EARTH_RADIUS * c


def meters_to_degrees(
    arr: Float[Array, "... 2"],
    lat: Float[Array, ""],
) -> Float[Array, "... 2"]:
    """Convert a [north, east] displacement in metres to [dlat, dlon] in degrees.

    lat is the reference latitude in degrees (used for the longitude scaling).
    """
    rad = arr / EARTH_RADIUS
    deg = jnp.degrees(rad)
    lon_scale = jnp.cos(jnp.radians(lat))
    return deg.at[..., 1].divide(lon_scale)


def degrees_to_meters(
    arr: Float[Array, "... 2"],
    lat: Float[Array, ""],
) -> Float[Array, "... 2"]:
    """Convert a [dlat, dlon] displacement in degrees to [north, east] in metres.

    lat is the reference latitude in degrees.
    """
    rad = jnp.radians(arr)
    meters = rad * EARTH_RADIUS
    lon_scale = jnp.cos(jnp.radians(lat))
    return meters.at[..., 1].multiply(lon_scale)
