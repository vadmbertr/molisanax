"""Bilinear interpolation on equally-spaced rectilinear A-grids."""

import jax.numpy as jnp

from ._types import Array, Float, Int

__all__ = [
    "linear_interp_1d",
    "bilinear_interp_2d",
    "spatiotemporal_interp",
]


def _index_and_weight(
    coords: Float[Array, "n"], x: Float[Array, ""]
) -> tuple[Int[Array, ""], Float[Array, ""]]:
    """Floor index and linear weight for a point on an equally-spaced 1-D grid.

    For grid spacing dx = coords[1] - coords[0] and origin x0 = coords[0]:
        i = clip(floor((x - x0) / dx), 0, n-2)
        w = (x - x0) / dx - i          (= fractional position within cell [i, i+1])

    w is in [0, 1) for in-range x; outside that range it extrapolates linearly
    (same behaviour as the previous searchsorted implementation).
    """
    x0 = coords[0]
    dx = coords[1] - coords[0]
    u = (x - x0) / dx
    i = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, coords.shape[0] - 2)
    w = u - i  # equivalent to (x - (x0 + i*dx)) / dx
    return i, w


def linear_interp_1d(
    values: Float[Array, "n"],
    coords: Float[Array, "n"],
    x: Float[Array, ""],
) -> Float[Array, ""]:
    """Linearly interpolate a 1-D field on an equally-spaced grid.

    Args:
        values: Field values at each grid node, shape ``(n,)``.
        coords: Equally-spaced 1-D grid coordinates, shape ``(n,)``.
        x: Query coordinate.

    Returns:
        Interpolated scalar value at ``x``. For ``x`` outside the grid the
        result is the linear extrapolation from the nearest cell.
    """
    i, w = _index_and_weight(coords, x)
    return values[i] * (1.0 - w) + values[i + 1] * w


def bilinear_interp_2d(
    values: Float[Array, "lat lon"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
) -> Float[Array, ""]:
    """Bilinearly interpolate a 2-D field on an equally-spaced rectilinear grid.

    Args:
        values: Field values, shape ``(n_lat, n_lon)``.
        lat_coords: Equally-spaced latitude coordinates in degrees, shape ``(n_lat,)``.
        lon_coords: Equally-spaced longitude coordinates in degrees, shape ``(n_lon,)``.
        lat: Query latitude in degrees.
        lon: Query longitude in degrees.

    Returns:
        Interpolated scalar value at ``(lat, lon)``.
    """
    il, wl = _index_and_weight(lat_coords, lat)
    jl, wj = _index_and_weight(lon_coords, lon)
    return (
        values[il,     jl    ] * (1.0 - wl) * (1.0 - wj)
        + values[il + 1, jl    ] * wl         * (1.0 - wj)
        + values[il,     jl + 1] * (1.0 - wl) * wj
        + values[il + 1, jl + 1] * wl         * wj
    )


def spatiotemporal_interp(
    values: Float[Array, "time lat lon"],
    t_coords: Float[Array, "time"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    t: Float[Array, ""],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
) -> Float[Array, ""]:
    """Trilinearly interpolate a field in time and space on an A-grid.

    Performs :func:`bilinear_interp_2d` at the two bounding time slices, then
    linearly blends the two results in time.

    Args:
        values: Field values, shape ``(n_time, n_lat, n_lon)``.
        t_coords: Equally-spaced time coordinates in seconds, shape ``(n_time,)``.
        lat_coords: Equally-spaced latitude coordinates in degrees.
        lon_coords: Equally-spaced longitude coordinates in degrees.
        t: Query time in seconds.
        lat: Query latitude in degrees.
        lon: Query longitude in degrees.

    Returns:
        Interpolated scalar value at ``(t, lat, lon)``.
    """
    it, wt = _index_and_weight(t_coords, t)
    v0 = bilinear_interp_2d(values[it],     lat_coords, lon_coords, lat, lon)
    v1 = bilinear_interp_2d(values[it + 1], lat_coords, lon_coords, lat, lon)
    return v0 * (1.0 - wt) + v1 * wt
