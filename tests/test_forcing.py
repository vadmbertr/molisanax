"""Tests for forcing.py: Field and Dataset."""

import jax
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


class TestFieldLonPeriodic:
    """Longitude wrap-around in Field.interp and Field.neighborhood."""

    def _periodic_field(self):
        # Global longitude grid: 4 cells of 90° spanning [0, 360)
        lats = jnp.array([0.0, 1.0])
        lons = jnp.array([0.0, 90.0, 180.0, 270.0])
        t_coords = jnp.array([0.0, 1.0])
        # values[t, lat, lon] encode the lon index directly
        slab = jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0, 3.0]), (2, 4))
        values = jnp.stack([slab, slab])  # (n_t=2, n_lat=2, n_lon=4)
        return Field(
            values=values,
            t_coords=t_coords,
            lat_coords=lats,
            lon_coords=lons,
            lon_period=360.0,
        )

    def test_interp_wraps(self):
        field = self._periodic_field()
        v = field.interp(jnp.array(0.5), jnp.array(0.0), jnp.array(315.0))
        # midpoint between lon-index 3 and lon-index 0
        assert float(v) == pytest.approx(1.5)

    def test_neighborhood_wraps_at_zero(self):
        field = self._periodic_field()
        # Centre on lon=0° (index 0), window=1 → indices [3, 0, 1]
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        assert patch.shape == (1, 1, 3)
        assert jnp.allclose(patch[0, 0], jnp.array([3.0, 0.0, 1.0]))

    def test_neighborhood_wraps_at_high_end(self):
        field = self._periodic_field()
        # Centre on lon=270° (index 3), window=1 → indices [2, 3, 0]
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(270.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        assert jnp.allclose(patch[0, 0], jnp.array([2.0, 3.0, 0.0]))

    def test_neighborhood_negative_lon_wraps(self):
        field = self._periodic_field()
        # -90° == 270° (index 3); window=1 → indices [2, 3, 0]
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(-90.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        assert jnp.allclose(patch[0, 0], jnp.array([2.0, 3.0, 0.0]))

    def test_non_periodic_neighborhood_clamps_at_zero(self):
        # Without lon_period the existing clamp behaviour is preserved
        lats = jnp.array([0.0, 1.0])
        lons = jnp.array([0.0, 90.0, 180.0, 270.0])
        t_coords = jnp.array([0.0, 1.0])
        slab = jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0, 3.0]), (2, 4))
        values = jnp.stack([slab, slab])
        field = Field(
            values=values,
            t_coords=t_coords,
            lat_coords=lats,
            lon_coords=lons,
        )
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        # clamp to start: indices [0, 1, 2]
        assert jnp.allclose(patch[0, 0], jnp.array([0.0, 1.0, 2.0]))

    def test_neighborhood_jit_compatible(self):
        field = self._periodic_field()

        @jax.jit
        def f(lon):
            return field.neighborhood(
                jnp.array(0.0), jnp.array(0.0), lon,
                t_window=0, lat_window=0, lon_window=1,
            )

        patch = f(jnp.array(0.0))
        assert jnp.allclose(patch[0, 0], jnp.array([3.0, 0.0, 1.0]))


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


class TestDatasetFromArrays:
    def _coords(self, n=4):
        t   = np.linspace(0.0, (n - 1) * 3600.0, n)
        lat = np.linspace(0.0, float(n - 1), n)
        lon = np.linspace(10.0, 10.0 + float(n - 1), n)
        return t, lat, lon

    def test_builds_dataset(self):
        t, lat, lon = self._coords()
        u = np.ones((4, 4, 4), dtype=np.float32)
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert "u" in dataset.fields
        assert isinstance(dataset["u"], Field)

    def test_interp_uniform_field(self):
        t, lat, lon = self._coords()
        u = np.ones((4, 4, 4), dtype=np.float32)
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        v = dataset["u"].interp(jnp.array(1800.0), jnp.array(1.5), jnp.array(11.5))
        assert float(v) == pytest.approx(1.0, abs=1e-5)

    def test_accepts_jax_arrays(self):
        t   = jnp.linspace(0.0, 3 * 3600.0, 4)
        lat = jnp.linspace(0.0, 3.0, 4)
        lon = jnp.linspace(10.0, 13.0, 4)
        u   = jnp.zeros((4, 4, 4))
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert dataset["u"].values.shape == (4, 4, 4)

    def test_lon_period_propagates_to_fields(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 90.0, 180.0, 270.0])
        u = np.broadcast_to(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                            (2, 2, 4))
        dataset = Dataset.from_arrays(
            {"u": u}, t=t, lat=lat, lon=lon, lon_period=360.0,
        )
        assert dataset["u"].lon_period == 360.0
        v = dataset["u"].interp(jnp.array(0.0), jnp.array(0.0), jnp.array(315.0))
        assert float(v) == pytest.approx(1.5)

    def test_from_arrays_and_from_xarray_agree(self):
        """from_arrays and from_xarray must produce identical field values."""
        ds = make_synthetic_ds()
        ds_dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        t = ds["time"].values.astype("datetime64[s]").astype(np.int64).astype(np.float32)
        arr_dataset = Dataset.from_arrays(
            {"u": ds["u"].values},
            t=t,
            lat=ds["lat"].values,
            lon=ds["lon"].values,
        )
        assert jnp.allclose(ds_dataset["u"].values, arr_dataset["u"].values)
