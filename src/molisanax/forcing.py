"""Forcing field representation and loading from xarray datasets or plain arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from ._types import Array, Float
from .interpolation import spatiotemporal_interp

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from jax import DTypeLike

__all__ = ["Field", "Dataset"]


def _nearest_idx(coords: Float[Array, "n"], x: Float[Array, ""], n: int) -> Array:
    """Nearest-neighbour index on an equally-spaced 1-D grid, clamped to [0, n-1]."""
    x0 = coords[0]
    dx = coords[1] - coords[0]
    return jnp.clip(jnp.round((x - x0) / dx).astype(jnp.int32), 0, n - 1)


class Field(eqx.Module):
    """A single scalar forcing field on a (time, lat, lon) rectilinear A-grid."""

    values: Float[Array, "time lat lon"]
    t_coords: Float[Array, "time"]
    lat_coords: Float[Array, "lat"]
    lon_coords: Float[Array, "lon"]

    def interp(
        self,
        t: Float[Array, ""],
        lat: Float[Array, ""],
        lon: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Interpolate the field at (t, lat, lon) using trilinear interpolation."""
        return spatiotemporal_interp(
            self.values, self.t_coords, self.lat_coords, self.lon_coords,
            t, lat, lon,
        )

    def neighborhood(
        self,
        t: Float[Array, ""],
        lat: Float[Array, ""],
        lon: Float[Array, ""],
        t_window: int = 1,
        lat_window: int = 1,
        lon_window: int = 1,
    ) -> Float[Array, "wt wlat wlon"]:
        """Extract a window of raw grid values centred on the nearest grid point.

        Args:
            t: Query time in seconds.
            lat: Query latitude in degrees.
            lon: Query longitude in degrees.
            t_window: Half-width along the time axis (window size = 2*t_window+1).
            lat_window: Half-width along the latitude axis.
            lon_window: Half-width along the longitude axis.

        Returns:
            Array of shape (2*t_window+1, 2*lat_window+1, 2*lon_window+1).
            The window is clamped to the grid boundary near the edges.
        """
        nt   = self.t_coords.shape[0]
        nlat = self.lat_coords.shape[0]
        nlon = self.lon_coords.shape[0]

        wt   = 2 * t_window   + 1
        wlat = 2 * lat_window + 1
        wlon = 2 * lon_window + 1

        it   = _nearest_idx(self.t_coords,   t,   nt)
        ilat = _nearest_idx(self.lat_coords, lat, nlat)
        ilon = _nearest_idx(self.lon_coords, lon, nlon)

        it_start   = jnp.clip(it   - t_window,   0, nt   - wt)
        ilat_start = jnp.clip(ilat - lat_window, 0, nlat - wlat)
        ilon_start = jnp.clip(ilon - lon_window, 0, nlon - wlon)

        return jax.lax.dynamic_slice(
            self.values,
            (it_start, ilat_start, ilon_start),
            (wt, wlat, wlon),
        )


class Dataset(eqx.Module):
    """Collection of forcing fields sharing the same (time, lat, lon) grid."""

    fields: dict[str, Field]

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def neighborhood(
        self,
        t: Float[Array, ""],
        lat: Float[Array, ""],
        lon: Float[Array, ""],
        t_window: int = 1,
        lat_window: int = 1,
        lon_window: int = 1,
    ) -> dict[str, Float[Array, "wt wlat wlon"]]:
        """Extract neighbourhoods for all fields. See Field.neighborhood for details."""
        return {
            name: field.neighborhood(t, lat, lon, t_window, lat_window, lon_window)
            for name, field in self.fields.items()
        }

    @staticmethod
    def from_arrays(
        fields: dict[str, Array],
        t: Array,
        lat: Array,
        lon: Array,
        dtype: DTypeLike = jnp.float32,
    ) -> Dataset:
        """Build a Dataset from numpy or JAX arrays.

        Args:
            fields: Mapping {field_name: array of shape (time, lat, lon)}.
            t: 1-D time coordinate array (seconds), equally spaced.
            lat: 1-D latitude coordinate array (degrees), equally spaced.
            lon: 1-D longitude coordinate array (degrees), equally spaced.
            dtype: JAX dtype for all arrays (default float32).

        Returns:
            Dataset with all fields on the given grid.
        """
        t_arr   = jnp.asarray(t,   dtype=dtype)
        lat_arr = jnp.asarray(lat, dtype=dtype)
        lon_arr = jnp.asarray(lon, dtype=dtype)
        loaded = {
            name: Field(
                values=jnp.asarray(v, dtype=dtype),
                t_coords=t_arr,
                lat_coords=lat_arr,
                lon_coords=lon_arr,
            )
            for name, v in fields.items()
        }
        return Dataset(fields=loaded)

    @staticmethod
    def from_xarray(
        ds: xr.Dataset,
        fields: dict[str, str],
        coordinates: dict[str, str],
        dtype: DTypeLike = jnp.float32,
    ) -> Dataset:
        """Load a Dataset from an xarray Dataset (zarr or netCDF backed).

        Args:
            ds: Source xarray Dataset.
            fields: Mapping {internal_name: xarray_variable_name}.
            coordinates: Mapping with keys "time", "lat", "lon" → xarray coord names.
            dtype: JAX dtype for all arrays (default float32).

        Returns:
            Dataset with all fields loaded into host memory as JAX arrays.
        """
        import numpy as np

        t_raw = ds[coordinates["time"]].values
        if hasattr(t_raw.dtype, "kind") and t_raw.dtype.kind == "M":
            t = t_raw.astype("datetime64[s]").astype(np.int64)
        else:
            t = t_raw

        field_arrays = {
            internal: ds[xr_name].values for internal, xr_name in fields.items()
        }
        return Dataset.from_arrays(
            field_arrays,
            t=t,
            lat=ds[coordinates["lat"]].values,
            lon=ds[coordinates["lon"]].values,
            dtype=dtype,
        )
