# molisanax

**Differentiable Lagrangian simulator for ocean surface trajectories, implemented in JAX/Equinox.**

`molisanax` integrates particle trajectories on the ocean surface by solving ODEs and SDEs over user-supplied forcing fields (e.g. surface currents). Every computation is fully differentiable via JAX automatic differentiation — both forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`) are supported.

## Project Status

**v0.1.0 — first iteration.** Core functionality is implemented and tested:
- Bilinear interpolation of rectilinear A-grid forcing fields
- Euler and Heun ODE/SDE solvers with constant step size
- Geographic unit conversions (metres ↔ degrees)
- Along-trajectory metrics (separation distance, normalised separation, Liu Index)
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

```python
import jax.numpy as jnp
from molisanax import solve_ode, Heun

# User-supplied term: f(t, y, args) -> velocity [dlat/dt, dlon/dt] in degrees/second
def my_term(t, y, args):
    dataset = args
    lat, lon = y[0], y[1]
    u = dataset["u"].interp(t, lat, lon)   # eastward velocity, m/s
    v = dataset["v"].interp(t, lat, lon)   # northward velocity, m/s
    from molisanax import meters_to_degrees
    return meters_to_degrees(jnp.array([v, u]), lat)

# Time axis in seconds (equally spaced)
ts = jnp.linspace(0.0, 86400.0 * 5, 121)  # 5 days, 1-hour steps

# Initial position [lat, lon] in degrees
y0 = jnp.array([48.0, -4.0])

# Integrate
trajectory = solve_ode(my_term, dataset, y0, ts, solver=Heun())
# trajectory: shape (121, 2) — [lat, lon] at each output time
```

### SDE simulation (stochastic ensemble)

```python
import jax.random as jr
from molisanax import solve_sde

def drift(t, y, args):
    ...  # same as ODE term

def diffusion(t, y, args):
    sigma = 1e-5  # noise amplitude in degrees/sqrt(s)
    return sigma * jnp.eye(2)  # shape (2, 2)

key = jr.key(0)
ensemble = solve_sde(drift, diffusion, dataset, y0, ts, key, n_samples=100, n_noise=2)
# ensemble: shape (100, 121, 2)
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

### Geographic conversions

```python
from molisanax import meters_to_degrees, degrees_to_meters

# Convert a [north, east] displacement in metres to [dlat, dlon] in degrees
disp_m = jnp.array([1000.0, 500.0])
lat_ref = jnp.array(45.0)
disp_deg = meters_to_degrees(disp_m, lat_ref)
```

### Differentiability

```python
import jax

# Reverse-mode gradient through the solver
loss = lambda y0: solve_ode(my_term, dataset, y0, ts).sum()
grad = jax.grad(loss)(y0)

# Forward-mode JVP
traj, tangent = jax.jvp(lambda y0: solve_ode(my_term, dataset, y0, ts), (y0,), (jnp.ones(2),))
```

### Trajectory metrics

```python
from molisanax import separation_distance, normalized_separation_distance, liu_index
import jax

# Single-trajectory metrics
sep = separation_distance(trajectory, reference_trajectory)     # shape (T,), metres
nsd = normalized_separation_distance(trajectory, reference)     # shape (T,), dimensionless
li  = liu_index(trajectory, reference)                         # shape (T,), dimensionless

# Ensemble metrics via vmap
ensemble_sep = jax.vmap(lambda y: separation_distance(y, reference))(ensemble)
# shape (n_samples, T)
```

## API Reference

### Solvers

| Function | Description |
|---|---|
| `solve_ode(term, args, y0, ts, solver, *, adjoint)` | Integrate ODE, return `(T, 2)` trajectory |
| `solve_sde(drift, diffusion, args, y0, ts, key, n_samples, n_noise, solver)` | Integrate SDE ensemble, return `(S, T, 2)` |

**Term signature**: `f(t: float_scalar, y: Float[Array, "2"], args: PyTree) -> Float[Array, "2"]`  
Return value is velocity `[dlat/dt, dlon/dt]` in **degrees per second**.

**Solvers**: `Euler()`, `Heun()` (default)

**Adjoint modes**: `"recursive_checkpoint"` (default, reverse-mode via `lax.scan`), `"forward"` (use with `jax.jvp`)

### Forcing

| Class | Description |
|---|---|
| `Field` | Scalar field on a `(time, lat, lon)` A-grid. Call `.interp(t, lat, lon)` to interpolate. |
| `Dataset` | Collection of `Field`s. Load via `Dataset.from_xarray(ds, fields, coordinates)`. |

### Geographic utilities

| Function | Description |
|---|---|
| `meters_to_degrees(arr, lat)` | Convert `[north, east]` metres → `[dlat, dlon]` degrees |
| `degrees_to_meters(arr, lat)` | Inverse of the above |
| `haversine(y1, y2)` | Great-circle distance in metres between two `[lat, lon]` points |

### Metrics

All metrics accept trajectories of shape `(T, 2)`. Use `jax.vmap` for ensemble shapes `(S, T, 2)`.

| Function | Formula |
|---|---|
| `separation_distance(y, y_ref)` | `haversine(y[t], y_ref[t])` at each t, in metres |
| `normalized_separation_distance(y, y_ref)` | `sep(t) / cumsum(arc_length_ref)[t]` |
| `liu_index(y, y_ref)` | `cumsum(sep)[t] / cumsum(arc_length_ref)[t]` |

## Design

`molisanax` has no diffrax or interpax dependency. The solver uses `jax.lax.scan` for the time-stepping loop, which JAX automatically differentiates in both forward and reverse mode. The `jax.checkpoint` decorator on the scan body provides memory-efficient reverse-mode AD (rematerialisation).

Forcing interpolation is a native JAX trilinear implementation (linear in time, bilinear in lat/lon) that is jit-compatible and differentiable with respect to position.

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
