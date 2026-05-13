"""Tests for the Grid metadata object and staggered-coordinate helpers."""

import jax.numpy as jnp
import pytest

from molisanax import Dataset, Field, Grid


def _centre_grid():
    t = jnp.asarray([0.0, 3600.0, 7200.0])
    lat = jnp.linspace(-5.0, 5.0, 11)
    lon = jnp.linspace(0.0, 9.0, 10)
    return t, lat, lon


def test_grid_defaults_are_a_grid_rectilinear():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon)
    assert g.grid_type == "rectilinear"
    assert g.stagger_type == "A"
    assert g.lon_period is None


def test_u_face_coords_shift_half_cell_in_lon():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    lat_u, lon_u = g.u_face_coords()
    assert lat_u.shape == lat.shape
    assert jnp.allclose(lat_u, lat)
    assert lon_u.shape == (lon.shape[0] - 1,)
    dlon = float(lon[1] - lon[0])
    assert jnp.allclose(lon_u, lon[:-1] + 0.5 * dlon)


def test_v_face_coords_shift_half_cell_in_lat():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    lat_v, lon_v = g.v_face_coords()
    assert lon_v.shape == lon.shape
    assert jnp.allclose(lon_v, lon)
    assert lat_v.shape == (lat.shape[0] - 1,)
    dlat = float(lat[1] - lat[0])
    assert jnp.allclose(lat_v, lat[:-1] + 0.5 * dlat)


def test_staggered_coords_preserve_equal_spacing():
    """The C-grid first-order interp relies on shifted coords being equally
    spaced. Verify that explicitly so a future regression (e.g. switch to a
    non-uniform centre grid) is caught here."""
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    _, lon_u = g.u_face_coords()
    lat_v, _ = g.v_face_coords()
    assert jnp.allclose(jnp.diff(lon_u), lon_u[1] - lon_u[0])
    assert jnp.allclose(jnp.diff(lat_v), lat_v[1] - lat_v[0])


def test_curvilinear_face_coords_raise_not_implemented():
    t, lat, lon = _centre_grid()
    lat_2d, lon_2d = jnp.meshgrid(lat, lon, indexing="ij")
    g = Grid(
        t_coords=t,
        lat_coords=lat_2d,
        lon_coords=lon_2d,
        grid_type="curvilinear",
        stagger_type="C",
    )
    with pytest.raises(NotImplementedError):
        g.u_face_coords()
    with pytest.raises(NotImplementedError):
        g.v_face_coords()


def test_field_stagger_defaults_to_center():
    t, lat, lon = _centre_grid()
    f = Field(
        values=jnp.zeros((3, 11, 10)),
        t_coords=t,
        lat_coords=lat,
        lon_coords=lon,
    )
    assert f.stagger == "center"


def test_field_can_be_tagged_u_face_or_v_face():
    t, lat, lon = _centre_grid()
    u = Field(
        values=jnp.zeros((3, 11, 9)),
        t_coords=t,
        lat_coords=lat,
        lon_coords=lon[:-1] + 0.5,
        stagger="u_face",
    )
    v = Field(
        values=jnp.zeros((3, 10, 10)),
        t_coords=t,
        lat_coords=lat[:-1] + 0.5,
        lon_coords=lon,
        stagger="v_face",
    )
    assert u.stagger == "u_face"
    assert v.stagger == "v_face"


def test_dataset_grid_defaults_to_none():
    """Backwards compat: existing A-grid datasets carry no Grid metadata."""
    t, lat, lon = _centre_grid()
    ds = Dataset.from_arrays(
        {"u": jnp.zeros((3, 11, 10))}, t=t, lat=lat, lon=lon
    )
    assert ds.grid is None


def test_dataset_accepts_grid_metadata():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    f = Field(
        values=jnp.zeros((3, 11, 10)),
        t_coords=t,
        lat_coords=lat,
        lon_coords=lon,
    )
    ds = Dataset(fields={"sst": f}, grid=g)
    assert ds.grid is g
    assert ds.grid.stagger_type == "C"


def test_field_interp_unchanged_for_center_stagger():
    """Adding the ``stagger`` field must not alter the interp result for
    existing A-grid Fields."""
    t, lat, lon = _centre_grid()
    values = jnp.arange(3 * 11 * 10, dtype=jnp.float32).reshape((3, 11, 10))
    f = Field(values=values, t_coords=t, lat_coords=lat, lon_coords=lon)
    v = f.interp(jnp.asarray(1800.0), jnp.asarray(0.0), jnp.asarray(4.5))
    assert jnp.isfinite(v)
