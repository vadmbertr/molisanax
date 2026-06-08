"""Grid metadata for Arakawa A- and C-grid forcing layouts.

The :class:`Grid` object describes the *centre* (tracer) coordinates of a
forcing dataset together with its stagger type. It is held as metadata on
:class:`pastax.Dataset` and acts as a coordinate-derivation helper for
C-grid loaders.

Interpolation on a C-grid is performed locally by each :class:`Field` using
its own (already-staggered) coordinates — the :class:`Grid` object itself is
not consulted at interp time. This keeps the existing ``Field.interp`` path
(bilinear on equally-spaced coords) valid for both A- and C-grid fields,
since shifting an equally-spaced grid by half a cell preserves equal spacing.

Curvilinear (2-D) coordinate arrays are accepted structurally for forward
compatibility but not yet supported by interpolation.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from ._types import Array, Float

__all__ = ["Grid"]


class Grid(eqx.Module):
    """Grid metadata: centre coordinates plus stagger and topology type.

    Attributes:
        t_coords: 1-D time coordinates in seconds, equally spaced.
        lat_coords: Latitude of cell centres. 1-D for rectilinear grids, 2-D
            ``(lat, lon)`` for curvilinear grids.
        lon_coords: Longitude of cell centres. 1-D for rectilinear grids,
            2-D ``(lat, lon)`` for curvilinear grids.
        grid_type: ``"rectilinear"`` (default) or ``"curvilinear"``.
            Curvilinear grids are accepted structurally but not yet supported
            by ``Field.interp``.
        stagger_type: ``"A"`` (default — all variables at cell centres) or
            ``"C"`` (NEMO-convention Arakawa C-grid: U on east faces, V on
            north faces).
        lon_period: If set (e.g. ``360.0``), longitude is treated as periodic
            with that period. The centre grid is assumed to span exactly one
            period.
    """

    t_coords: Float[Array, "time"]
    lat_coords: Float[Array, "..."]
    lon_coords: Float[Array, "..."]
    grid_type: Literal["rectilinear", "curvilinear"] = eqx.field(
        static=True, default="rectilinear"
    )
    stagger_type: Literal["A", "C"] = eqx.field(static=True, default="A")
    lon_period: float | None = eqx.field(static=True, default=None)

    def u_face_coords(self) -> tuple[Float[Array, "lat"], Float[Array, "lon_u"]]:
        r"""Return ``(lat_u, lon_u)`` — coordinates of U-face centres (NEMO C-grid).

        For a centre grid of size ``nlon`` in longitude, U lives on the
        ``nlon - 1`` east faces between adjacent cells:

        .. math::

            \mathrm{lon}_u[i] = \tfrac{1}{2}\left(\mathrm{lon}_c[i] + \mathrm{lon}_c[i+1]\right)

        Latitude is unchanged: :math:`\mathrm{lat}_u = \mathrm{lat}_c`.

        Raises:
            NotImplementedError: For curvilinear grids.
        """
        self._check_rectilinear("u_face_coords")
        lon_c = self.lon_coords
        lon_u = 0.5 * (lon_c[:-1] + lon_c[1:])
        return self.lat_coords, lon_u

    def v_face_coords(self) -> tuple[Float[Array, "lat_v"], Float[Array, "lon"]]:
        r"""Return ``(lat_v, lon_v)`` — coordinates of V-face centres (NEMO C-grid).

        For a centre grid of size ``nlat`` in latitude, V lives on the
        ``nlat - 1`` north faces between adjacent cells:

        .. math::

            \mathrm{lat}_v[j] = \tfrac{1}{2}\left(\mathrm{lat}_c[j] + \mathrm{lat}_c[j+1]\right)

        Longitude is unchanged: :math:`\mathrm{lon}_v = \mathrm{lon}_c`.

        Raises:
            NotImplementedError: For curvilinear grids.
        """
        self._check_rectilinear("v_face_coords")
        lat_c = self.lat_coords
        lat_v = 0.5 * (lat_c[:-1] + lat_c[1:])
        return lat_v, self.lon_coords

    def _check_rectilinear(self, method: str) -> None:
        if self.grid_type != "rectilinear":
            raise NotImplementedError(
                f"Grid.{method} is only implemented for rectilinear grids "
                f"(got grid_type={self.grid_type!r})."
            )
        if jnp.ndim(self.lat_coords) != 1 or jnp.ndim(self.lon_coords) != 1:
            raise ValueError(
                f"Rectilinear Grid expected 1-D lat/lon coords, got "
                f"lat ndim={jnp.ndim(self.lat_coords)}, "
                f"lon ndim={jnp.ndim(self.lon_coords)}."
            )
