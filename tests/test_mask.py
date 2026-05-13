"""Tests for the optional land-mask plumbing on Field and the loaders.

PR 1 of the coastal-robustness iteration. No interpolation behaviour is
changed by this PR; these tests verify only:

- ``Field.mask`` defaults to ``None`` (backwards-compatible PyTree).
- All four loaders accept a ``masks`` kwarg.
- When the source contains NaN, a mask is auto-inferred and NaN is
  replaced with 0 in the stored ``values``.
- When the source is NaN-free and no user mask is supplied, the
  resulting ``Field.mask`` stays ``None`` (preserving the legacy path).
- ``Field.interp`` with ``mask=None`` produces the same output as before
  (verified by the existing 137 tests + an explicit equality check
  against a NaN-free input here).
"""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from molisanax import Dataset, Field


def test_field_mask_defaults_to_none():
    f = Field(
        values=jnp.zeros((2, 3, 4)),
        t_coords=jnp.asarray([0.0, 1.0]),
        lat_coords=jnp.linspace(0.0, 2.0, 3),
        lon_coords=jnp.linspace(0.0, 3.0, 4),
    )
    assert f.mask is None


def test_field_accepts_explicit_mask():
    mask = jnp.array([[False, True], [True, False]])
    f = Field(
        values=jnp.zeros((1, 2, 2)),
        t_coords=jnp.asarray([0.0]),
        lat_coords=jnp.asarray([0.0, 1.0]),
        lon_coords=jnp.asarray([0.0, 1.0]),
        mask=mask,
    )
    assert f.mask is not None
    assert jnp.array_equal(f.mask, mask)


class TestFromArraysMask:
    def _coords(self):
        t = np.linspace(0.0, 7200.0, 3)
        lat = np.linspace(0.0, 3.0, 4)
        lon = np.linspace(10.0, 14.0, 5)
        return t, lat, lon

    def test_nan_free_input_leaves_mask_none(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert ds["u"].mask is None

    def test_nan_input_auto_infers_mask(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        u[:, 0, 0] = np.nan       # one land cell
        u[1, 2, 3] = np.nan       # NaN at one timestep only — still land
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)

        assert ds["u"].mask is not None
        assert ds["u"].mask.shape == (4, 5)
        assert bool(ds["u"].mask[0, 0]) is True
        assert bool(ds["u"].mask[2, 3]) is True
        # An ocean cell stays False
        assert bool(ds["u"].mask[1, 1]) is False

    def test_nan_values_replaced_with_zero(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        u[:, 0, 0] = np.nan
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert bool(jnp.isnan(ds["u"].values).any()) is False
        assert float(ds["u"].values[0, 0, 0]) == 0.0

    def test_user_mask_overrides_inference(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        u[:, 0, 0] = np.nan       # would be inferred as land
        explicit = np.zeros((4, 5), dtype=bool)
        explicit[3, 4] = True     # different cell than the NaN
        ds = Dataset.from_arrays(
            {"u": u}, t=t, lat=lat, lon=lon, masks={"u": explicit},
        )
        # User mask wins; the NaN-derived cell is NOT marked as land.
        assert bool(ds["u"].mask[0, 0]) is False
        assert bool(ds["u"].mask[3, 4]) is True
        # NaN was still cleared from values regardless.
        assert bool(jnp.isnan(ds["u"].values).any()) is False

    def test_per_field_mask_routing(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        v = np.ones((3, 4, 5), dtype=np.float32)
        u_mask = np.zeros((4, 5), dtype=bool); u_mask[0, 0] = True
        ds = Dataset.from_arrays(
            {"u": u, "v": v}, t=t, lat=lat, lon=lon, masks={"u": u_mask},
        )
        # u gets the explicit mask, v stays None (no NaN, no user mask).
        assert ds["u"].mask is not None and bool(ds["u"].mask[0, 0]) is True
        assert ds["v"].mask is None

    def test_rejects_3d_mask(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        bad = np.zeros((3, 4, 5), dtype=bool)
        with pytest.raises(ValueError, match=r"expected 2-D bool array"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, masks={"u": bad})

    def test_rejects_wrong_2d_shape(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        bad = np.zeros((4, 4), dtype=bool)
        with pytest.raises(ValueError, match=r"expected 2-D bool array"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, masks={"u": bad})


class TestFromXarrayMask:
    def _ds_with_nan_land(self):
        t = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        lat = np.linspace(0.0, 2.0, 3)
        lon = np.linspace(10.0, 13.0, 4)
        u = np.ones((2, 3, 4), dtype=np.float32)
        u[:, 0, 0] = np.nan
        return xr.Dataset(
            {"u": (["time", "lat", "lon"], u)},
            coords={"time": t, "lat": lat, "lon": lon},
        )

    def test_nan_in_xarray_auto_infers_mask(self):
        ds_xr = self._ds_with_nan_land()
        ds = Dataset.from_xarray(
            ds_xr,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert ds["u"].mask is not None
        assert bool(ds["u"].mask[0, 0]) is True
        assert bool(jnp.isnan(ds["u"].values).any()) is False

    def test_explicit_mask_via_from_xarray(self):
        ds_xr = self._ds_with_nan_land()
        explicit = np.zeros((3, 4), dtype=bool)
        explicit[2, 3] = True
        ds = Dataset.from_xarray(
            ds_xr,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
            masks={"u": explicit},
        )
        # User mask wins over NaN inference.
        assert bool(ds["u"].mask[0, 0]) is False
        assert bool(ds["u"].mask[2, 3]) is True


class TestFromArraysCGridMask:
    def _inputs(self, nlat=5, nlon=6, nt=2):
        t = np.linspace(0.0, 3600.0, nt)
        lat = np.linspace(0.0, float(nlat - 1), nlat)
        lon = np.linspace(0.0, float(nlon - 1), nlon)
        u = np.ones((nt, nlat, nlon - 1), dtype=np.float32)
        v = np.ones((nt, nlat - 1, nlon), dtype=np.float32)
        return t, lat, lon, u, v

    def test_nan_in_u_infers_u_face_mask(self):
        t, lat, lon, u, v = self._inputs()
        u[:, 0, 0] = np.nan
        ds = Dataset.from_arrays_cgrid(t, lat, lon, u, v)
        assert ds["u"].mask is not None
        assert ds["u"].mask.shape == (5, 5)  # (nlat, nlon - 1)
        assert bool(ds["u"].mask[0, 0]) is True
        assert ds["v"].mask is None

    def test_nan_in_v_infers_v_face_mask(self):
        t, lat, lon, u, v = self._inputs()
        v[:, 0, 0] = np.nan
        ds = Dataset.from_arrays_cgrid(t, lat, lon, u, v)
        assert ds["u"].mask is None
        assert ds["v"].mask is not None
        assert ds["v"].mask.shape == (4, 6)  # (nlat - 1, nlon)

    def test_nan_in_tracer_infers_centre_mask(self):
        t, lat, lon, u, v = self._inputs()
        sst = np.full((2, 5, 6), 15.0, dtype=np.float32)
        sst[:, 0, 0] = np.nan
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, u, v, tracers={"sst": sst},
        )
        assert ds["sst"].mask is not None
        assert ds["sst"].mask.shape == (5, 6)
        assert bool(ds["sst"].mask[0, 0]) is True

    def test_explicit_u_face_mask_with_correct_shape(self):
        t, lat, lon, u, v = self._inputs()
        u_mask = np.zeros((5, 5), dtype=bool)
        u_mask[2, 3] = True
        ds = Dataset.from_arrays_cgrid(t, lat, lon, u, v, masks={"u": u_mask})
        assert bool(ds["u"].mask[2, 3]) is True

    def test_explicit_u_mask_wrong_shape_rejected(self):
        t, lat, lon, u, v = self._inputs()
        # Centre shape instead of u-face shape
        wrong = np.zeros((5, 6), dtype=bool)
        with pytest.raises(ValueError, match=r"masks\['u'\]"):
            Dataset.from_arrays_cgrid(t, lat, lon, u, v, masks={"u": wrong})

    def test_explicit_v_mask_wrong_shape_rejected(self):
        t, lat, lon, u, v = self._inputs()
        wrong = np.zeros((5, 6), dtype=bool)
        with pytest.raises(ValueError, match=r"masks\['v'\]"):
            Dataset.from_arrays_cgrid(t, lat, lon, u, v, masks={"v": wrong})

    def test_explicit_tracer_mask_routed_correctly(self):
        t, lat, lon, u, v = self._inputs()
        sst = np.full((2, 5, 6), 15.0, dtype=np.float32)
        tr_mask = np.zeros((5, 6), dtype=bool)
        tr_mask[1, 1] = True
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, u, v,
            tracers={"sst": sst}, masks={"sst": tr_mask},
        )
        assert bool(ds["sst"].mask[1, 1]) is True
        # u/v still mask-less (no NaN, no user mask)
        assert ds["u"].mask is None
        assert ds["v"].mask is None


class TestFromXarrayCGridMask:
    def test_nan_in_xarray_cgrid_infers_mask(self):
        t = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.linspace(0.0, 5.0, 6)
        u_data = np.ones((2, 5, 5), dtype=np.float32)
        v_data = np.ones((2, 4, 6), dtype=np.float32)
        u_data[:, 0, 0] = np.nan
        ds_xr = xr.Dataset(
            {
                "uo": (["time", "lat", "lon_u"], u_data),
                "vo": (["time", "lat_v", "lon"], v_data),
            },
            coords={
                "time": t, "lat": lat, "lon": lon,
                "lon_u": lon[:-1] + 0.5 * (lon[1] - lon[0]),
                "lat_v": lat[:-1] + 0.5 * (lat[1] - lat[0]),
            },
        )
        ds = Dataset.from_xarray_cgrid(
            ds_xr,
            u_name="uo", v_name="vo",
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert ds["u"].mask is not None
        assert bool(ds["u"].mask[0, 0]) is True
        assert ds["v"].mask is None
        assert bool(jnp.isnan(ds["u"].values).any()) is False


def test_interp_unchanged_when_mask_none():
    """For NaN-free input the mask path is fully skipped and Field.interp
    produces the same output as before this PR."""
    t = np.linspace(0.0, 3600.0, 3)
    lat = np.linspace(0.0, 4.0, 5)
    lon = np.linspace(0.0, 5.0, 6)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((3, 5, 6)).astype(np.float32)

    ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
    assert ds["u"].mask is None
    val = float(ds["u"].interp(jnp.asarray(1800.0), jnp.asarray(2.3), jnp.asarray(3.1)))
    # Build a Field directly without going through the loader — should match.
    f_direct = Field(
        values=jnp.asarray(u),
        t_coords=jnp.asarray(t, dtype=jnp.float32),
        lat_coords=jnp.asarray(lat, dtype=jnp.float32),
        lon_coords=jnp.asarray(lon, dtype=jnp.float32),
    )
    val_direct = float(
        f_direct.interp(jnp.asarray(1800.0), jnp.asarray(2.3), jnp.asarray(3.1))
    )
    assert val == pytest.approx(val_direct, abs=1e-6)
