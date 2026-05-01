"""Forcing field representation and loading from xarray datasets."""

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
            t_window: Half-width of the window along the time axis (window size = 2*t_window+1).
            lat_window: Half-width along the latitude axis.
            lon_window: Half-width along the longitude axis.

        Returns:
            Array of shape (2*t_window+1, 2*lat_window+1, 2*lon_window+1) containing
            raw field values in the neighbourhood. The window is clamped to the grid
            boundary when the query point is near the edge.
        """
        nt = self.t_coords.shape[0]
        nlat = self.lat_coords.shape[0]
        nlon = self.lon_coords.shape[0]

        wt = 2 * t_window + 1
        wlat = 2 * lat_window + 1
        wlon = 2 * lon_window + 1

        # Nearest-neighbour index along each axis
        it = jnp.clip(jnp.searchsorted(self.t_coords, t), 0, nt - 1)
        ilat = jnp.clip(jnp.searchsorted(self.lat_coords, lat), 0, nlat - 1)
        ilon = jnp.clip(jnp.searchsorted(self.lon_coords, lon), 0, nlon - 1)

        # Window start index (clamped so the slice stays within bounds)
        it_start = jnp.clip(it - t_window, 0, nt - wt)
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
    def from_xarray(
        ds: xr.Dataset,
        fields: dict[str, str],
        coordinates: dict[str, str],
        dtype: DTypeLike = jnp.float32,
    ) -> Dataset:
        """Load a Dataset from an xarray Dataset.

        Args:
            ds: Source xarray Dataset (zarr or netCDF backed).
            fields: Mapping {internal_name: xarray_variable_name}.
            coordinates: Mapping with keys "time", "lat", "lon" → xarray coord names.
            dtype: JAX dtype for all arrays (default float32).

        Returns:
            Dataset with all fields loaded into host memory as JAX arrays.
        """
        import numpy as np

        t_raw = ds[coordinates["time"]].values
        if hasattr(t_raw.dtype, "kind") and t_raw.dtype.kind == "M":
            t_arr = jnp.asarray(t_raw.astype("datetime64[s]").astype(np.int64), dtype=dtype)
        else:
            t_arr = jnp.asarray(t_raw, dtype=dtype)

        lat_arr = jnp.asarray(ds[coordinates["lat"]].values, dtype=dtype)
        lon_arr = jnp.asarray(ds[coordinates["lon"]].values, dtype=dtype)

        loaded: dict[str, Field] = {}
        for internal_name, xr_name in fields.items():
            values = jnp.asarray(ds[xr_name].values, dtype=dtype)
            loaded[internal_name] = Field(
                values=values,
                t_coords=t_arr,
                lat_coords=lat_arr,
                lon_coords=lon_arr,
            )

        return Dataset(fields=loaded)
