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
    u_data = np.ones((3, 3, 3), dtype=np.float32)  # uniform u=1
    v_data = np.zeros((3, 3, 3), dtype=np.float32)
    ds = xr.Dataset(
        {"u": (["time", "lat", "lon"], u_data),
         "v": (["time", "lat", "lon"], v_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


class TestField:
    def test_interp_returns_scalar(self):
        lats = jnp.array([0.0, 1.0, 2.0])
        lons = jnp.array([10.0, 11.0, 12.0])
        t_coords = jnp.array([0.0, 86400.0, 172800.0])
        values = jnp.ones((3, 3, 3))
        field = Field(values=values, t_coords=t_coords, lat_coords=lats, lon_coords=lons)
        v = field.interp(jnp.array(43200.0), jnp.array(1.0), jnp.array(11.0))
        assert v.shape == ()
        assert float(v) == pytest.approx(1.0)


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
        # Uniform u=1 everywhere, interpolation should return 1.0
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
