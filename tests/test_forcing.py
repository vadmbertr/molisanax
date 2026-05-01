"""Tests for forcing.py: Field and Dataset."""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from molisanax.forcing import Dataset, Field


def make_synthetic_ds():
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([10.0, 11.0, 12.0])
    u_data = np.ones((3, 3, 3), dtype=np.float32)
    v_data = np.zeros((3, 3, 3), dtype=np.float32)
    return xr.Dataset(
        {"u": (["time", "lat", "lon"], u_data),
         "v": (["time", "lat", "lon"], v_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def make_field(n=5, values_fill=1.0):
    lats = jnp.linspace(0.0, 4.0, n)
    lons = jnp.linspace(10.0, 14.0, n)
    t_coords = jnp.linspace(0.0, 4 * 86400.0, n)
    values = jnp.full((n, n, n), values_fill)
    return Field(values=values, t_coords=t_coords, lat_coords=lats, lon_coords=lons)


class TestField:
    def test_interp_returns_scalar(self):
        field = make_field()
        v = field.interp(jnp.array(43200.0), jnp.array(1.0), jnp.array(11.0))
        assert v.shape == ()
        assert float(v) == pytest.approx(1.0)

    def test_neighborhood_shape_default(self):
        field = make_field(n=7)
        patch = field.neighborhood(
            jnp.array(3 * 86400.0), jnp.array(2.0), jnp.array(12.0)
        )
        assert patch.shape == (3, 3, 3)  # 2*1+1 in each dim

    def test_neighborhood_shape_custom(self):
        field = make_field(n=9)
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(10.0),
            t_window=2, lat_window=1, lon_window=3,
        )
        assert patch.shape == (5, 3, 7)

    def test_neighborhood_values_uniform_field(self):
        field = make_field(n=7, values_fill=3.14)
        patch = field.neighborhood(
            jnp.array(3 * 86400.0), jnp.array(2.0), jnp.array(12.0)
        )
        assert jnp.allclose(patch, jnp.full_like(patch, 3.14))

    def test_neighborhood_clamped_at_boundary(self):
        # Query at the very start — window should still have the right shape
        field = make_field(n=7)
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(10.0)
        )
        assert patch.shape == (3, 3, 3)


class TestDataset:
    def test_from_xarray_loads_fields(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u", "v": "v"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert "u" in dataset.fields
        assert "v" in dataset.fields

    def test_from_xarray_interp_uniform_field(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        field = dataset["u"]
        t_s = float(np.datetime64("2020-01-01T12:00:00", "s").astype(np.int64))
        v = field.interp(jnp.array(t_s, dtype=jnp.float32), jnp.array(1.0), jnp.array(11.0))
        assert float(v) == pytest.approx(1.0, abs=1e-5)

    def test_getitem(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert isinstance(dataset["u"], Field)

    def test_neighborhood_returns_dict(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u", "v": "v"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        # Use a large-enough grid: ds has 3 points per axis, window=1 → need 3 points → OK
        patches = dataset.neighborhood(
            jnp.array(float(np.datetime64("2020-01-02", "s").astype(np.int64)), dtype=jnp.float32),
            jnp.array(1.0),
            jnp.array(11.0),
        )
        assert set(patches.keys()) == {"u", "v"}
        assert patches["u"].shape == (3, 3, 3)
