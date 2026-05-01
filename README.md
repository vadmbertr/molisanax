# molisanax

**Differentiable Lagrangian simulator for ocean surface trajectories, implemented in JAX/Equinox.**

`molisanax` integrates particle trajectories on the ocean surface by solving ODEs and SDEs over user-supplied forcing fields (e.g. surface currents). Every computation is fully differentiable via JAX automatic differentiation — both forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`) are supported.

## Project Status

**v0.1.0 — second iteration.** Core functionality implemented and tested (73 tests):
- Bilinear interpolation of rectilinear A-grid forcing fields, with neighbourhood extraction
- Unified `solve()` function — automatically detects ODE vs SDE from the term's return type
- Euler and Heun solvers with constant step size; noise drawn internally for SDE
- Geographic unit conversions (metres ↔ degrees)
- Along-trajectory metrics with optional ensemble (vmap) mode
- xarray (zarr/netCDF) dataset loading

See [`docs/project_status.md`](docs/project_status.md) for current capabilities and known limitations.

## Installation

```bash
pip install molisanax                      # core (JAX, Equinox, jaxtyping)
pip install "molisanax[forcing]"           # + xarray, zarr, netCDF4
```

From source:

```bash
git clone https://github.com/vadmbertr/molisanax
cd molisanax
pip install -e ".[dev]"
```

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

A term returning `(drift, noise_amplitude)` triggers SDE mode. The solver draws
`z ~ N(0, I₂)` at each step and computes `dy = (drift + noise_amplitude * z) * dt`.

```python
import jax.random as jr

def my_term(t, y, args):
    dataset = args
    lat, lon = y[0], y[1]
    u = dataset["u"].interp(t, lat, lon)
    v = dataset["v"].interp(t, lat, lon)
    drift = meters_to_degrees(jnp.array([v, u]), lat)
    noise_amplitude = jnp.full(2, 1e-5)   # deg/s noise scale per component
    return drift, noise_amplitude

key = jr.key(0)

# Single stochastic trajectory
traj = solve(my_term, dataset, y0, ts, key=key)
# shape (121, 2)

# Ensemble of 100 independent realisations
ensemble = solve(my_term, dataset, y0, ts, key=key, n_samples=100)
# shape (100, 121, 2)
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

## API Reference

### Solver

```
solve(term, args, y0, ts, solver=Heun(), *, key=None, n_samples=None, adjoint="recursive_checkpoint")
```

| Term return type | Mode | Output shape |
|---|---|---|
| `Float[Array, "2"]` | ODE | `(T, 2)` |
| `(Float[Array, "2"], Float[Array, "2"])` | SDE (`key` required) | `(T, 2)` if `n_samples=None`, else `(S, T, 2)` |

**Term signature**: `f(t: scalar, y: Float[Array, "2"], args: PyTree) -> ...`
Returns velocity `[dlat/dt, dlon/dt]` in **degrees per second** (ODE) or `(drift, noise_amplitude)` both in deg/s (SDE).

**Solvers**: `Euler()`, `Heun()` (default)

### Forcing

| | |
|---|---|
| `Field.interp(t, lat, lon)` | Trilinear interpolation → scalar |
| `Field.neighborhood(t, lat, lon, t_window, lat_window, lon_window)` | Raw grid patch via `lax.dynamic_slice` |
| `Dataset.from_xarray(ds, fields, coordinates, dtype)` | Load from xarray Dataset |
| `Dataset.neighborhood(...)` | Neighbourhood for all fields → `dict[str, Array]` |

### Metrics

All accept `ensemble=False` (single trajectory) or `ensemble=True` (auto-vmaps over first axis).

| | |
|---|---|
| `separation_distance(y, y_ref)` | Haversine at each step, metres |
| `normalized_separation_distance(y, y_ref)` | `sep(t) / cumsum_ref_arc(t)` |
| `liu_index(y, y_ref)` | `cumsum_sep(t) / cumsum_ref_arc(t)` |

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
