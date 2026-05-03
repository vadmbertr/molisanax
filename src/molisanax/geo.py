"""Geographic utilities: unit conversions and Haversine distance."""

import jax.numpy as jnp

from ._safe_math import safe_sqrt
from ._types import Array, Float

__all__ = [
    "EARTH_RADIUS",
    "haversine",
    "meters_to_degrees",
    "degrees_to_meters",
]

EARTH_RADIUS: float = 6_371_008.8
"""Mean Earth radius in metres (IUGG 2015 mean radius)."""


def haversine(
    y1: Float[Array, "2"],
    y2: Float[Array, "2"],
) -> Float[Array, ""]:
    """Great-circle distance between two ``[lat, lon]`` points.

    Uses the spherical haversine formula with :data:`EARTH_RADIUS` as the
    sphere radius.

    Args:
        y1: First point ``[lat, lon]`` in degrees.
        y2: Second point ``[lat, lon]`` in degrees.

    Returns:
        Great-circle distance in metres.
    """
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
    """Convert a ``[north, east]`` displacement in metres to ``[dlat, dlon]`` in degrees.

    Uses a flat-Earth approximation around ``lat``: the meridional component
    is converted via ``EARTH_RADIUS``; the zonal component is additionally
    divided by ``cos(lat)`` to account for shrinking longitude circles toward
    the poles.

    Args:
        arr: Displacement(s) ``[north, east]`` in metres. The last axis must
            have size 2; leading axes are passed through unchanged.
        lat: Reference latitude in degrees, used for the longitude scaling.

    Returns:
        Same shape as ``arr``, but expressed as ``[dlat, dlon]`` in degrees.
    """
    rad = arr / EARTH_RADIUS
    deg = jnp.degrees(rad)
    lon_scale = jnp.cos(jnp.radians(lat))
    return deg.at[..., 1].divide(lon_scale)


def degrees_to_meters(
    arr: Float[Array, "... 2"],
    lat: Float[Array, ""],
) -> Float[Array, "... 2"]:
    """Convert a ``[dlat, dlon]`` displacement in degrees to ``[north, east]`` in metres.

    Inverse of :func:`meters_to_degrees`. Uses a flat-Earth approximation
    around ``lat``.

    Args:
        arr: Displacement(s) ``[dlat, dlon]`` in degrees. The last axis must
            have size 2; leading axes are passed through unchanged.
        lat: Reference latitude in degrees, used for the longitude scaling.

    Returns:
        Same shape as ``arr``, but expressed as ``[north, east]`` in metres.
    """
    rad = jnp.radians(arr)
    meters = rad * EARTH_RADIUS
    lon_scale = jnp.cos(jnp.radians(lat))
    return meters.at[..., 1].multiply(lon_scale)
