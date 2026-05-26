# molisanax

**Differentiable Lagrangian simulator for ocean surface trajectories, implemented in JAX/Equinox.**

`molisanax` integrates particle trajectories on the ocean surface by solving ODEs and SDEs over user-supplied forcing fields (e.g. surface currents). Every computation is fully differentiable via JAX automatic differentiation — both forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`) are supported.

📖 **Documentation:** <https://vadmbertr.github.io/molisanax/> — full API reference and a runnable [tutorial notebook](docs/tutorial.ipynb).

## Project Status

- Bilinear interpolation of rectilinear forcing fields, with neighbourhood extraction
- A-grid and NEMO-convention Arakawa C-grid forcing layouts (`Dataset.from_arrays_cgrid` / `from_xarray_cgrid`)
- Coastal robustness on A-grid: NaN-inferred land masks, inverse-distance partial-cell bilinear, and an opt-in partial-slip scheme via `Dataset.velocity_interp`
- Unified `solve()` function — ODE/SDE mode selected by caller (no introspection)
- Euler, Heun and RK4 solvers; SDE term receives noise vector `z` directly for full flexibility
- Forward or backwards-in-time integration (pass an increasing or decreasing `ts`)
- Geographic unit conversions (metres ↔ degrees)
- Along-trajectory metrics with optional ensemble (vmap) mode
- (Proper) Scoring rules to evaluate stochastic simulators
- xarray (zarr/netCDF) dataset loading; also `Dataset.from_arrays` for plain numpy/JAX arrays

See [`docs/project_status.md`](docs/project_status.md) for current capabilities and known limitations.

## Installation

From Git;

```bash
pip install git+https://github.com/vadmbertr/molisanax             # core (JAX, Equinox, jaxtyping)
pip install "git+https://github.com/vadmbertr/molisanax[forcing]"  # + xarray, zarr, netCDF4
```

From source:

```bash
git clone https://github.com/vadmbertr/molisanax
cd molisanax
pip install -e ".[dev]"
```

Installing a JAX **GPU version** should be done prior to installing `molisanax`, following [https://docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html).

## Quick Start

### ODE simulation (deterministic)

A term returns a single velocity array — `solve()` detects ODE mode automatically.

```python
import jax.numpy as jnp
from molisanax import solve, Heun, meters_to_degrees

def my_term(t, y, args):
    dataset = args
    lat, lon = y[0], y[1]
    u = dataset["u"].interp(t, lat, lon)   # eastward velocity, m/s
    v = dataset["v"].interp(t, lat, lon)   # northward velocity, m/s
    return meters_to_degrees(jnp.array([v, u]), lat)  # → deg/s

ts = jnp.linspace(0.0, 86400.0 * 5, 121)  # 5 days, 1-hour steps
y0 = jnp.array([48.0, -4.0])              # [lat, lon] in degrees

trajectory = solve(my_term, dataset, y0, ts, Heun())
# trajectory: shape (121, 2)
```

### SDE simulation (stochastic ensemble)

SDE mode is activated by passing `key`, `noise`, or `n_noise` to `solve()`.
The term receives the pre-sampled noise vector `z` as a fourth argument and
returns the full velocity — no `(f, g)` decomposition is imposed.

```python
import jax.random as jr

def my_term(t, y, args, z):
    dataset = args
    lat, lon = y[0], y[1]
    u = dataset["u"].interp(t, lat, lon)
    v = dataset["v"].interp(t, lat, lon)
    drift = meters_to_degrees(jnp.array([v, u]), lat)
    noise_amplitude = jnp.full(2, 1e-5)   # deg/s noise scale per component
    return drift + noise_amplitude * z

key = jr.key(0)

# Single stochastic trajectory (z has dimension 2)
traj = solve(my_term, dataset, y0, ts, key=key, n_noise=2)
# shape (121, 2)

# Ensemble of 100 independent realisations
ensemble = solve(my_term, dataset, y0, ts, key=key, n_noise=2, n_samples=100)
# shape (100, 121, 2)

# Pre-sampled noise (reproducible; n_noise inferred from noise.shape[-1])
n_steps = ts.shape[0] - 1
noise = jr.normal(key, shape=(n_steps, 2))
traj = solve(my_term, dataset, y0, ts, noise=noise)
# shape (121, 2)
```

### Loading forcing fields from xarray

```python
import xarray as xr
from molisanax import Dataset

ds = xr.open_zarr("path/to/currents.zarr")
dataset = Dataset.from_xarray(
    ds,
    fields={"u": "uo", "v": "vo"},
    coordinates={"time": "time", "lat": "latitude", "lon": "longitude"},
)
```

Or directly from numpy/JAX arrays:

```python
import numpy as np

t   = np.linspace(0.0, 4 * 86400.0, 5)  # seconds
lat = np.linspace(40.0, 50.0, 100)
lon = np.linspace(-10.0, 0.0, 100)
u_data = np.ones((5, 100, 100), dtype=np.float32)

dataset = Dataset.from_arrays({"u": u_data}, t=t, lat=lat, lon=lon)
```

### Loading C-grid forcing (NEMO convention)

For data on an Arakawa C-grid, U lives on the east faces of the centre cells
(shape `(time, nlat, nlon - 1)`) and V on the north faces (shape
`(time, nlat - 1, nlon)`). `from_arrays_cgrid` auto-derives the staggered
coordinates as half-cell shifts of the centre grid:

```python
from molisanax import Dataset

dataset = Dataset.from_arrays_cgrid(
    t, center_lat, center_lon,
    u_values,                       # (T, nlat, nlon - 1)  on east faces
    v_values,                       # (T, nlat - 1, nlon)  on north faces
    tracers={"sst": sst_values},    # optional, at cell centres (T, nlat, nlon)
)
# dataset["u"].stagger == "u_face"
# dataset["v"].stagger == "v_face"
# dataset.grid.stagger_type == "C"
```

The same `term` you wrote for A-grid forcing works unchanged — each `Field`
stores its own (already-shifted) coordinates, so `Field.interp` applies the
correct bilinear-on-shifted-coords sample at the particle position.

xarray analogue:

```python
dataset = Dataset.from_xarray_cgrid(
    ds,
    u_name="uo", v_name="vo",
    coordinates={"time": "time", "lat": "lat", "lon": "lon"},  # centre coords
    tracers={"sst": "thetao"},
)
```

### Coastal forcing

Real ocean forcing has land. By default the loaders detect land cells
automatically:

```python
# u_data has NaN at every land cell (CMEMS / CF convention)
dataset = Dataset.from_arrays({"u": u_data, "v": v_data}, t=t, lat=lat, lon=lon)
# NaN was replaced with 0 in the stored values; a 2-D bool mask
# was inferred from the NaN locations and attached to each Field:
# dataset["u"].mask.shape == (nlat, nlon)
```

If your data marks land with zeros (NEMO convention) or a custom flag,
pass an explicit mask instead:

```python
land_mask = (raw_bathy == 0)               # True where land
dataset = Dataset.from_arrays(
    {"u": u_data, "v": v_data}, t=t, lat=lat, lon=lon,
    masks={"u": land_mask, "v": land_mask},
)
```

The mask is consumed by `Field.interp` to switch from naive bilinear to
**inverse-distance partial-cell weighting** whenever a cell straddles
the coast: land corners are dropped and the remaining ocean corners are
weighted by `1 / d²` from the query point. Fully land-bound cells
return `0`. This eliminates the "stuck particle" artefact that plagues
naive bilinear interpolation on A-grid coastal data — particles
released near a coast slide along it at the correct ocean velocity
instead of stalling.

For richer wall-physics control, use `Dataset.velocity_interp` to
interpolate `(U, V)` jointly with an opt-in partial-slip correction:

```python
def my_term(t, y, args):
    dataset = args
    vel = dataset.velocity_interp(t, y[0], y[1], scheme="partialslip")
    return meters_to_degrees(vel, y[0])   # vel is [v, u] = [dlat/dt, dlon/dt]
```

`scheme="default"` (the default) composes per-field `Field.interp`
(inverse-distance when a mask is present). `scheme="partialslip"`
applies a tunable wall-slip correction near fully-land edges:
`U` near a latitudinal coast is rescaled by `(slip_a + slip_b * wl)`,
and `V` near a longitudinal coast by `(slip_a + slip_b * wj)`. The
default `slip_a = slip_b = 0.5` gives a half-slip wall; `slip_a = 1,
slip_b = 0` recovers full free-slip. Partial-slip is A-grid only —
calling it on a C-grid dataset raises `NotImplementedError`.

C-grid forcing handles coasts correctly without any mask, **provided
U and V at land-adjacent faces are exactly zero** (the NEMO output
convention): the face-normal velocity then vanishes at the coast by
construction.

All three coastal paths (inverse-distance, partial-slip, naive
bilinear) are gradient-safe under `jax.grad` and `jax.jvp`.

### Neighbourhood extraction

```python
# Extract a 5×5×5 patch of raw grid values around a query point
patch = dataset["u"].neighborhood(t, lat, lon, t_window=2, lat_window=2, lon_window=2)
# shape (5, 5, 5)

# Or for all fields at once
patches = dataset.neighborhood(t, lat, lon, lat_window=1, lon_window=1)
# dict[str, Array] with shape (3, 3, 3) per field
```

### Geographic conversions

```python
from molisanax import meters_to_degrees, degrees_to_meters

disp_m = jnp.array([1000.0, 500.0])  # [north, east] metres
lat_ref = jnp.array(45.0)
disp_deg = meters_to_degrees(disp_m, lat_ref)   # [dlat, dlon] degrees
```

### Backwards-in-time integration

Pass a strictly decreasing `ts` to integrate backwards. All solvers (`Euler`, `Heun`, `RK4`) handle this transparently because `dt = ts[1] - ts[0]` becomes negative:

```python
y0_end = jnp.array([48.0, -4.0])
ts_bwd = jnp.linspace(86400.0 * 5, 0.0, 121)   # 5 days, backwards
backtrack = solve(my_term, dataset, y0_end, ts_bwd, RK4())
# backtrack[-1] is the source position 5 days earlier.
```

### Differentiability

```python
import jax

# Reverse-mode gradient through the ODE solver
grad = jax.grad(lambda y0: solve(my_ode_term, dataset, y0, ts).sum())(y0)

# Forward-mode JVP
traj, tangent = jax.jvp(lambda y0: solve(my_ode_term, dataset, y0, ts), (y0,), (jnp.ones(2),))
```

### Trajectory metrics

```python
from molisanax import separation_distance, normalized_separation_distance, liu_index

# Single-trajectory metrics
sep = separation_distance(trajectory, reference)          # (T,), metres
nsd = normalized_separation_distance(trajectory, reference)  # (T,), dimensionless
li  = liu_index(trajectory, reference)                    # (T,), dimensionless

# Ensemble metrics — vmap handled automatically
sep_ens = separation_distance(ensemble, reference, ensemble=True)  # (S, T)
li_ens  = liu_index(ensemble, reference, ensemble=True)            # (S, T)
```

### (Proper) Scoring rules

```python
from molisanax import dawid_sebastiani, energy_score, squared_error, variogram_score

# Along trajectory scores
ds_ts = dawid_sebastiani(ens, ref, reduce=None)  # (T,)
es_ts = energy_score(ens, ref, reduce=None)  # (T,)
se_ts = squared_error(ens, ref, reduce=None)  # (T,)
vs_ts = variogram_score(ens, ref, reduce=None)  # (T,)

# Final scores
ds_t1 = dawid_sebastiani(ens, ref, reduce="last")  # scalar
es_t1 = energy_score(ens, ref, reduce="last")  # scalar
se_t1 = squared_error(ens, ref, reduce="last")  # scalar
vs_t1 = variogram_score(ens, ref, reduce="last")  # scalar

# Aggregated scores
ds_agg = dawid_sebastiani(ens, ref, reduce="sum")  # scalar
es_agg = energy_score(ens, ref, reduce="sum")  # scalar
se_agg = squared_error(ens, ref, reduce="sum")  # scalar
vs_agg = variogram_score(ens, ref, reduce="sum")  # scalar

from molisanax import haversine

# Custom score kernel (relevant for the energy score and the square error only)
es_agg = energy_score(ens, ref, kernel=haversine)
se_agg = squared_error(ens, ref, kernel=haversine)
```

## API Reference

The full API reference — every public symbol, signature, and docstring — lives on the documentation site: <https://vadmbertr.github.io/molisanax/api>.

## Design

No diffrax or interpax dependency. The integration loop uses `jax.lax.scan` with `jax.checkpoint` on the body for memory-efficient reverse-mode AD. ODE/SDE mode is detected by probing the term with `jax.eval_shape` before entering jit.

## Running Tests

```bash
pytest -q
```

## Dependencies

- [JAX](https://github.com/google/jax) ≥ 0.4.30
- [Equinox](https://github.com/patrick-kidger/equinox) ≥ 0.11.0
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping) ≥ 0.2.30
- xarray, zarr, netCDF4 (optional, for forcing loading)

## License

MIT
