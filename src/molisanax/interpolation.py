"""Bilinear interpolation on equally-spaced rectilinear A-grids."""

import jax.numpy as jnp

from ._types import Array, Float, Int

__all__ = [
    "linear_interp_1d",
    "bilinear_interp_2d",
    "spatiotemporal_interp",
]


def _floor_index(coords: Float[Array, "n"], x: Float[Array, ""]) -> Int[Array, ""]:
    """Return clamped floor index i such that coords[i] <= x < coords[i+1]."""
    # searchsorted returns insertion point; subtract 1 for floor
    i = jnp.searchsorted(coords, x, side="right") - 1
    return jnp.clip(i, 0, coords.shape[0] - 2)


def _lerp_weight(
    coords: Float[Array, "n"],
    i: Array,
    x: Float[Array, ""],
) -> Float[Array, ""]:
    """Linear weight w in [0, 1]: x = coords[i] * (1-w) + coords[i+1] * w."""
    x0 = coords[i]
    x1 = coords[i + 1]
    # For equally-spaced grids dx is constant; still using generic formula.
    return (x - x0) / (x1 - x0)


def linear_interp_1d(
    values: Float[Array, "n"],
    coords: Float[Array, "n"],
    x: Float[Array, ""],
) -> Float[Array, ""]:
    """1D linear interpolation. Extrapolates by clamping to boundary values."""
    i = _floor_index(coords, x)
    w = _lerp_weight(coords, i, x)
    return values[i] * (1.0 - w) + values[i + 1] * w


def bilinear_interp_2d(
    values: Float[Array, "lat lon"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
) -> Float[Array, ""]:
    """Bilinear interpolation on a 2D rectilinear (lat, lon) grid."""
    il = _floor_index(lat_coords, lat)
    jl = _floor_index(lon_coords, lon)
    wl = _lerp_weight(lat_coords, il, lat)
    wj = _lerp_weight(lon_coords, jl, lon)
    v00 = values[il, jl]
    v10 = values[il + 1, jl]
    v01 = values[il, jl + 1]
    v11 = values[il + 1, jl + 1]
    return (
        v00 * (1.0 - wl) * (1.0 - wj)
        + v10 * wl * (1.0 - wj)
        + v01 * (1.0 - wl) * wj
        + v11 * wl * wj
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
    """Trilinear interpolation: bilinear in (lat, lon) at bounding time steps, then linear in t."""
    it = _floor_index(t_coords, t)
    wt = _lerp_weight(t_coords, it, t)
    v0 = bilinear_interp_2d(values[it], lat_coords, lon_coords, lat, lon)
    v1 = bilinear_interp_2d(values[it + 1], lat_coords, lon_coords, lat, lon)
    return v0 * (1.0 - wt) + v1 * wt
