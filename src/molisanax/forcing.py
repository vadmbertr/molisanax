"""Forcing field representation and loading from xarray datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
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


class Dataset(eqx.Module):
    """Collection of forcing fields sharing the same (time, lat, lon) grid."""

    fields: dict[str, Field]

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

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
