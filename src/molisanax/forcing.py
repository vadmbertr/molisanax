"""Forcing field representation and loading from xarray datasets or plain arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ._types import Array, Float
from .grid import Grid
from .interpolation import spatiotemporal_interp

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from jax import DTypeLike

__all__ = ["Field", "Dataset"]


def _coerce_time_to_seconds(t: Array) -> Array:
    """Convert a datetime64 time coordinate to int seconds; pass-through otherwise.

    Both ``Dataset.from_arrays`` and ``Dataset.from_xarray`` route through this
    helper, so users may pass NumPy ``datetime64`` arrays (any unit) directly:
    they are reinterpreted as seconds since the Unix epoch. Plain numeric arrays
    are returned unchanged and are treated as "seconds since some reference"
    (only differences matter to the solver).
    """
    import numpy as np

    t_arr = np.asarray(t)
    if t_arr.dtype.kind == "M":
        return t_arr.astype("datetime64[s]").astype(np.int64)
    return t_arr


def _nearest_idx(coords: Float[Array, "n"], x: Float[Array, ""], n: int) -> Array:
    """Nearest-neighbour index on an equally-spaced 1-D grid, clamped to [0, n-1]."""
    x0 = coords[0]
    dx = coords[1] - coords[0]
    return jnp.clip(jnp.round((x - x0) / dx).astype(jnp.int32), 0, n - 1)


def _nearest_idx_periodic(
    coords: Float[Array, "n"], x: Float[Array, ""], n: int, period: float
) -> Array:
    """Nearest-neighbour index on a periodic equally-spaced 1-D grid (mod n)."""
    x0 = coords[0]
    dx = coords[1] - coords[0]
    return jnp.round(((x - x0) % period) / dx).astype(jnp.int32) % n


class Field(eqx.Module):
    """A single scalar forcing field on a (time, lat, lon) rectilinear grid.

    Attributes:
        values: Field values, shape ``(time, lat, lon)``.
        t_coords: 1-D time coordinates in seconds, equally spaced.
        lat_coords: 1-D latitude coordinates in degrees, equally spaced.
        lon_coords: 1-D longitude coordinates in degrees, equally spaced.
        lon_period: If set (e.g. ``360.0``), the longitude axis is treated as
            periodic with that period in both ``interp`` and ``neighborhood``.
            The grid is assumed to span exactly one period: the cell at
            ``lon_coords[-1] + dlon`` is identified with ``lon_coords[0]``.
            ``None`` (default) means no wrapping.
        stagger: Position of this field on the parent grid. ``"center"``
            (default) is the A-grid / tracer position; ``"u_face"`` and
            ``"v_face"`` mark the eastern and northern velocity faces of a
            NEMO-convention Arakawa C-grid. The coordinate arrays
            (``lat_coords``, ``lon_coords``) must already describe the
            stagger position — they are what ``interp`` consults — so a
            C-grid ``Field`` with ``stagger="u_face"`` carries
            half-cell-shifted longitudes. Treat this attribute as metadata
            for downstream code that needs to distinguish velocity faces
            from centres; ``Field.interp`` itself is the same bilinear
            scheme regardless.
    """

    values: Float[Array, "time lat lon"]
    t_coords: Float[Array, "time"]
    lat_coords: Float[Array, "lat"]
    lon_coords: Float[Array, "lon"]
    lon_period: float | None = eqx.field(static=True, default=None)
    stagger: Literal["center", "u_face", "v_face"] = eqx.field(
        static=True, default="center"
    )

    def interp(
        self,
        t: Float[Array, ""],
        lat: Float[Array, ""],
        lon: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Trilinearly interpolate the field at a single ``(t, lat, lon)`` point.

        Args:
            t: Query time in seconds.
            lat: Query latitude in degrees.
            lon: Query longitude in degrees.

        Returns:
            Interpolated scalar value at the query point. Outside the grid the
            interpolation extrapolates linearly (clamping to grid boundaries
            beyond one cell). When ``lon_period`` is set, longitude wraps
            instead of extrapolating.
        """
        return spatiotemporal_interp(
            self.values, self.t_coords, self.lat_coords, self.lon_coords,
            t, lat, lon, lon_period=self.lon_period,
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
            Time and latitude windows are clamped to the grid boundary near the
            edges. The longitude window wraps modulo ``lon_period`` when that
            attribute is set, otherwise it is clamped like the others.
        """
        nt   = self.t_coords.shape[0]
        nlat = self.lat_coords.shape[0]
        nlon = self.lon_coords.shape[0]

        wt   = 2 * t_window   + 1
        wlat = 2 * lat_window + 1
        wlon = 2 * lon_window + 1

        it   = _nearest_idx(self.t_coords,   t,   nt)
        ilat = _nearest_idx(self.lat_coords, lat, nlat)

        it_start   = jnp.clip(it   - t_window,   0, nt   - wt)
        ilat_start = jnp.clip(ilat - lat_window, 0, nlat - wlat)

        if self.lon_period is None:
            ilon = _nearest_idx(self.lon_coords, lon, nlon)
            ilon_start = jnp.clip(ilon - lon_window, 0, nlon - wlon)
            return jax.lax.dynamic_slice(
                self.values,
                (it_start, ilat_start, ilon_start),
                (wt, wlat, wlon),
            )

        ilon = _nearest_idx_periodic(self.lon_coords, lon, nlon, self.lon_period)
        block = jax.lax.dynamic_slice(
            self.values,
            (it_start, ilat_start, 0),
            (wt, wlat, nlon),
        )
        lon_idx = (ilon - lon_window + jnp.arange(wlon)) % nlon
        return block[:, :, lon_idx]


class Dataset(eqx.Module):
    """Collection of named :class:`Field` instances sharing a common grid.

    Attributes:
        fields: Mapping ``{field_name: Field}``. For A-grid datasets every
            field lives at cell centres; for C-grid datasets velocity
            fields live on their respective faces (see :attr:`Field.stagger`).
        grid: Optional :class:`Grid` metadata describing the centre
            coordinates and stagger type of the underlying ocean grid.
            ``None`` (default) keeps the legacy A-grid path unchanged.
    """

    fields: dict[str, Field]
    grid: Grid | None = None

    def __getitem__(self, name: str) -> Field:
        """Return the :class:`Field` registered under ``name``.

        Args:
            name: Field name as registered in ``fields``.

        Returns:
            The :class:`Field` instance.
        """
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
        """Extract a neighbourhood patch from every field at one query point.

        Equivalent to calling :meth:`Field.neighborhood` on every field with
        the same query and window arguments. Useful for SDE terms that need
        local spatial gradients (e.g. Smagorinsky-style diffusion).

        Args:
            t: Query time in seconds.
            lat: Query latitude in degrees.
            lon: Query longitude in degrees.
            t_window: Half-width along the time axis (window size = ``2*t_window+1``).
            lat_window: Half-width along the latitude axis.
            lon_window: Half-width along the longitude axis.

        Returns:
            Mapping ``{field_name: array}`` where each array has shape
            ``(2*t_window+1, 2*lat_window+1, 2*lon_window+1)``.
        """
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
        lon_period: float | None = None,
    ) -> Dataset:
        """Build a Dataset from numpy or JAX arrays.

        Args:
            fields: Mapping {field_name: array of shape (time, lat, lon)}.
            t: 1-D time coordinate array. Either equally-spaced numeric values
                (seconds since an arbitrary reference) or a NumPy ``datetime64``
                array (any unit); the latter is auto-converted to int seconds
                since the Unix epoch.
            lat: 1-D latitude coordinate array (degrees), equally spaced.
            lon: 1-D longitude coordinate array (degrees), equally spaced.
            dtype: JAX dtype for all arrays (default float32).
            lon_period: If set (e.g. ``360.0``), all fields are constructed
                with periodic longitude wrapping. The grid must span exactly
                one period.

        Returns:
            Dataset with all fields on the given grid.
        """
        t = _coerce_time_to_seconds(t)
        t_arr   = jnp.asarray(t,   dtype=dtype)
        lat_arr = jnp.asarray(lat, dtype=dtype)
        lon_arr = jnp.asarray(lon, dtype=dtype)
        loaded = {
            name: Field(
                values=jnp.asarray(v, dtype=dtype),
                t_coords=t_arr,
                lat_coords=lat_arr,
                lon_coords=lon_arr,
                lon_period=lon_period,
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
        lon_period: float | None = None,
    ) -> Dataset:
        """Load a Dataset from an xarray Dataset (zarr or netCDF backed).

        Args:
            ds: Source xarray Dataset.
            fields: Mapping {internal_name: xarray_variable_name}.
            coordinates: Mapping with keys "time", "lat", "lon" → xarray coord names.
            dtype: JAX dtype for all arrays (default float32).
            lon_period: If set (e.g. ``360.0``), all fields are constructed
                with periodic longitude wrapping. The grid must span exactly
                one period.

        Returns:
            Dataset with all fields loaded into host memory as JAX arrays.
        """
        field_arrays = {
            internal: ds[xr_name].values for internal, xr_name in fields.items()
        }
        return Dataset.from_arrays(
            field_arrays,
            t=ds[coordinates["time"]].values,
            lat=ds[coordinates["lat"]].values,
            lon=ds[coordinates["lon"]].values,
            dtype=dtype,
            lon_period=lon_period,
        )
