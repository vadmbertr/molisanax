"""Forcing field representation and loading from xarray datasets or plain arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ._types import Array, Bool, Float
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
        mask: Optional 2-D boolean land mask aligned with ``(lat, lon)``;
            ``True`` marks a land cell, ``False`` marks ocean. Assumed
            time-invariant (wet-and-dry is out of scope). ``None`` (default)
            means no land logic — ``Field.interp`` is plain bilinear. When
            a mask is present, coastal interpolation schemes (added in a
            later iteration) consult it to drop land corners.
    """

    values: Float[Array, "time lat lon"]
    t_coords: Float[Array, "time"]
    lat_coords: Float[Array, "lat"]
    lon_coords: Float[Array, "lon"]
    lon_period: float | None = eqx.field(static=True, default=None)
    stagger: Literal["center", "u_face", "v_face"] = eqx.field(
        static=True, default="center"
    )
    mask: Bool[Array, "lat lon"] | None = None

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
            instead of extrapolating. When ``self.mask`` is set, coastal
            cells use inverse-distance partial-cell weighting and fully
            land-bound cells return ``0`` (see
            :func:`molisanax.interpolation.bilinear_interp_2d`).
        """
        return spatiotemporal_interp(
            self.values, self.t_coords, self.lat_coords, self.lon_coords,
            t, lat, lon, lon_period=self.lon_period, mask=self.mask,
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
        masks: dict[str, Array] | None = None,
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
            masks: Optional ``{field_name: 2-D bool array of shape (lat, lon)}``
                land masks. ``True`` marks a land cell. When a field appears
                in ``masks``, that mask is used. Otherwise a mask is inferred
                from NaN locations in the values array (collapsed across the
                time axis). Fields with neither user-supplied nor inferred
                NaN entries carry ``mask=None`` — interp behaviour is then
                bit-exact identical to the legacy mask-less path. NaN values
                in the input are always replaced with 0 in the stored
                ``values`` so no NaN can leak into interpolation.

        Returns:
            Dataset with all fields on the given grid.
        """
        t = _coerce_time_to_seconds(t)
        t_arr   = jnp.asarray(t,   dtype=dtype)
        lat_arr = jnp.asarray(lat, dtype=dtype)
        lon_arr = jnp.asarray(lon, dtype=dtype)
        nlat = int(lat_arr.shape[0])
        nlon = int(lon_arr.shape[0])
        masks = masks or {}
        loaded: dict[str, Field] = {}
        for name, v in fields.items():
            v_arr = jnp.asarray(v, dtype=dtype)
            clean, mask = _resolve_mask(
                v_arr, masks.get(name),
                expected_mask_shape=(nlat, nlon),
                field_name=name,
            )
            loaded[name] = Field(
                values=clean,
                t_coords=t_arr,
                lat_coords=lat_arr,
                lon_coords=lon_arr,
                lon_period=lon_period,
                mask=mask,
            )
        return Dataset(fields=loaded)

    @staticmethod
    def from_xarray(
        ds: xr.Dataset,
        fields: dict[str, str],
        coordinates: dict[str, str],
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
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
            masks: Optional land masks keyed by internal field name; see
                :meth:`from_arrays` for semantics. If omitted, masks are
                inferred from NaN — which matches the CMEMS / CF
                ``_FillValue`` convention.

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
            masks=masks,
        )

    @staticmethod
    def from_arrays_cgrid(
        t: Array,
        center_lat: Array,
        center_lon: Array,
        u_values: Array,
        v_values: Array,
        tracers: dict[str, Array] | None = None,
        *,
        u_lat: Array | None = None,
        u_lon: Array | None = None,
        v_lat: Array | None = None,
        v_lon: Array | None = None,
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
    ) -> Dataset:
        """Build a Dataset on a NEMO-convention Arakawa C-grid.

        The centre grid ``(center_lat, center_lon)`` carries any tracer
        fields. U lives on the east faces of the centre cells (one fewer
        longitude column) and V lives on the north faces (one fewer
        latitude row). When the staggered coordinate arrays are omitted
        they are auto-derived from the centre grid as half-cell shifts
        (see :meth:`Grid.u_face_coords` / :meth:`Grid.v_face_coords`).

        Args:
            t: 1-D time coordinates (seconds or NumPy ``datetime64``).
            center_lat: 1-D centre latitudes (degrees), equally spaced.
            center_lon: 1-D centre longitudes (degrees), equally spaced.
            u_values: U-component values, shape ``(time, nlat, nlon - 1)``.
            v_values: V-component values, shape ``(time, nlat - 1, nlon)``.
            tracers: Optional mapping ``{name: array of shape (time, nlat, nlon)}``
                for additional fields at cell centres.
            u_lat: Override for U latitudes (defaults to ``center_lat``).
            u_lon: Override for U longitudes (defaults to centre lons shifted
                east by half a cell, length ``nlon - 1``).
            v_lat: Override for V latitudes (defaults to centre lats shifted
                north by half a cell, length ``nlat - 1``).
            v_lon: Override for V longitudes (defaults to ``center_lon``).
            dtype: JAX dtype for all arrays (default float32).
            lon_period: If set (e.g. ``360.0``), the centre grid is treated
                as periodic in longitude. Tracer fields receive
                ``lon_period``; U/V faces do not (their coordinate arrays
                no longer span a full period, so periodic wrapping would be
                ill-defined at first order).
            masks: Optional land masks keyed by field name (``"u"``, ``"v"``,
                or any tracer name). Each mask is a 2-D bool array; the
                expected shape per field is
                ``(nlat, nlon - 1)`` for ``"u"``,
                ``(nlat - 1, nlon)`` for ``"v"``,
                and ``(nlat, nlon)`` for tracers. When a field is absent
                from ``masks``, a mask is inferred from NaN locations in
                that field's values array. NaN values are always replaced
                with 0 in the stored ``values``.

        Returns:
            Dataset with ``fields={"u": Field(stagger="u_face"),
            "v": Field(stagger="v_face"), **tracers}`` and a C-grid
            :class:`Grid` metadata object.
        """
        t = _coerce_time_to_seconds(t)
        t_arr   = jnp.asarray(t,           dtype=dtype)
        lat_arr = jnp.asarray(center_lat,  dtype=dtype)
        lon_arr = jnp.asarray(center_lon,  dtype=dtype)

        nt   = int(t_arr.shape[0])
        nlat = int(lat_arr.shape[0])
        nlon = int(lon_arr.shape[0])

        u_arr = jnp.asarray(u_values, dtype=dtype)
        v_arr = jnp.asarray(v_values, dtype=dtype)
        _check_cgrid_shape("u_values", u_arr.shape, (nt, nlat, nlon - 1))
        _check_cgrid_shape("v_values", v_arr.shape, (nt, nlat - 1, nlon))

        grid = Grid(
            t_coords=t_arr,
            lat_coords=lat_arr,
            lon_coords=lon_arr,
            grid_type="rectilinear",
            stagger_type="C",
            lon_period=lon_period,
        )
        derived_u_lat, derived_u_lon = grid.u_face_coords()
        derived_v_lat, derived_v_lon = grid.v_face_coords()

        u_lat_arr = jnp.asarray(u_lat, dtype=dtype) if u_lat is not None else derived_u_lat
        u_lon_arr = jnp.asarray(u_lon, dtype=dtype) if u_lon is not None else derived_u_lon
        v_lat_arr = jnp.asarray(v_lat, dtype=dtype) if v_lat is not None else derived_v_lat
        v_lon_arr = jnp.asarray(v_lon, dtype=dtype) if v_lon is not None else derived_v_lon
        _check_cgrid_shape("u_lat", u_lat_arr.shape, (nlat,))
        _check_cgrid_shape("u_lon", u_lon_arr.shape, (nlon - 1,))
        _check_cgrid_shape("v_lat", v_lat_arr.shape, (nlat - 1,))
        _check_cgrid_shape("v_lon", v_lon_arr.shape, (nlon,))

        masks = masks or {}
        u_clean, u_mask = _resolve_mask(
            u_arr, masks.get("u"),
            expected_mask_shape=(nlat, nlon - 1), field_name="u",
        )
        v_clean, v_mask = _resolve_mask(
            v_arr, masks.get("v"),
            expected_mask_shape=(nlat - 1, nlon), field_name="v",
        )
        loaded: dict[str, Field] = {
            "u": Field(
                values=u_clean, t_coords=t_arr,
                lat_coords=u_lat_arr, lon_coords=u_lon_arr,
                lon_period=None, stagger="u_face", mask=u_mask,
            ),
            "v": Field(
                values=v_clean, t_coords=t_arr,
                lat_coords=v_lat_arr, lon_coords=v_lon_arr,
                lon_period=None, stagger="v_face", mask=v_mask,
            ),
        }
        if tracers:
            for name, arr in tracers.items():
                a = jnp.asarray(arr, dtype=dtype)
                _check_cgrid_shape(f"tracers[{name!r}]", a.shape, (nt, nlat, nlon))
                tr_clean, tr_mask = _resolve_mask(
                    a, masks.get(name),
                    expected_mask_shape=(nlat, nlon), field_name=name,
                )
                loaded[name] = Field(
                    values=tr_clean, t_coords=t_arr,
                    lat_coords=lat_arr, lon_coords=lon_arr,
                    lon_period=lon_period, stagger="center", mask=tr_mask,
                )
        return Dataset(fields=loaded, grid=grid)

    @staticmethod
    def from_xarray_cgrid(
        ds: xr.Dataset,
        *,
        u_name: str,
        v_name: str,
        coordinates: dict[str, str],
        tracers: dict[str, str] | None = None,
        staggered_coordinates: dict[str, str] | None = None,
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
    ) -> Dataset:
        """Load a C-grid Dataset from an xarray Dataset.

        Centre coordinates (used for time and tracer fields) come from
        ``coordinates``; staggered U/V coordinates are auto-derived from
        the centre grid as half-cell shifts unless overridden via
        ``staggered_coordinates``.

        Args:
            ds: Source xarray Dataset.
            u_name: xarray variable name for the U component
                (shape ``(time, nlat, nlon - 1)``).
            v_name: xarray variable name for the V component
                (shape ``(time, nlat - 1, nlon)``).
            coordinates: Mapping with keys ``"time"``, ``"lat"``, ``"lon"``
                → xarray coord names for the centre grid.
            tracers: Optional ``{internal_name: xarray_variable_name}`` for
                extra centre-grid fields.
            staggered_coordinates: Optional override mapping with any
                subset of keys ``"u_lat"``, ``"u_lon"``, ``"v_lat"``,
                ``"v_lon"`` → xarray coord names. Unspecified keys are
                auto-derived.
            dtype: JAX dtype for all arrays (default float32).
            lon_period: Forwarded to :meth:`from_arrays_cgrid`.
            masks: Forwarded to :meth:`from_arrays_cgrid` (per-field
                ``{"u", "v", <tracer_name>}`` keys).

        Returns:
            Dataset with C-grid stagger and :class:`Grid` metadata.
        """
        stag = staggered_coordinates or {}
        return Dataset.from_arrays_cgrid(
            t=ds[coordinates["time"]].values,
            center_lat=ds[coordinates["lat"]].values,
            center_lon=ds[coordinates["lon"]].values,
            u_values=ds[u_name].values,
            v_values=ds[v_name].values,
            tracers={
                internal: ds[xr_name].values
                for internal, xr_name in (tracers or {}).items()
            } or None,
            u_lat=ds[stag["u_lat"]].values if "u_lat" in stag else None,
            u_lon=ds[stag["u_lon"]].values if "u_lon" in stag else None,
            v_lat=ds[stag["v_lat"]].values if "v_lat" in stag else None,
            v_lon=ds[stag["v_lon"]].values if "v_lon" in stag else None,
            dtype=dtype,
            lon_period=lon_period,
            masks=masks,
        )


def _check_cgrid_shape(name: str, got: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if got != expected:
        raise ValueError(
            f"C-grid shape mismatch for {name}: expected {expected}, got {got}. "
            f"NEMO convention requires U at shape (time, nlat, nlon-1) and "
            f"V at shape (time, nlat-1, nlon)."
        )


def _resolve_mask(
    values: Float[Array, "time lat lon"],
    user_mask: Array | None,
    *,
    expected_mask_shape: tuple[int, int],
    field_name: str,
) -> tuple[Float[Array, "time lat lon"], Bool[Array, "lat lon"] | None]:
    """Replace NaN in ``values`` with 0 and resolve the field's land mask.

    Rules (in order):

    1. NaN values are always replaced with 0 in the returned values array.
    2. If ``user_mask`` is provided, it is validated against
       ``expected_mask_shape`` and used as-is. Time-varying (3-D) masks are
       rejected.
    3. Otherwise, if NaN was present in ``values``, infer a 2-D mask from
       ``isnan(values).any(axis=0)``.
    4. If no user mask and no NaN, return ``mask=None`` — the resulting
       Field is bit-exact identical (PyTree structure included) to one
       built before the mask feature was added.

    Returns:
        ``(clean_values, mask)`` where ``mask`` is either a 2-D bool array
        or ``None``.
    """
    nan_locs = jnp.isnan(values)
    clean_values = jnp.where(nan_locs, jnp.asarray(0.0, dtype=values.dtype), values)

    if user_mask is not None:
        mask = jnp.asarray(user_mask, dtype=jnp.bool_)
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"masks[{field_name!r}]: expected 2-D bool array of shape "
                f"{expected_mask_shape}, got {mask.shape}. Time-varying masks "
                "(wet-and-dry) are not supported."
            )
        return clean_values, mask

    if bool(nan_locs.any()):
        inferred = nan_locs.any(axis=0)
        return clean_values, inferred

    return clean_values, None
