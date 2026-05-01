# molisanax — Implementation Plan

*Date: 2026-05-01*

---

## 0. Scope and Non-Goals

This plan covers the **first iteration** of `molisanax`: a pip-installable, fully differentiable Lagrangian simulator for ocean surface trajectories implemented in JAX/Equinox.

**In scope:**
- Rectilinear A-grid forcing (equally spaced lat/lon/time), (bi)linear interpolation
- Euler and Heun ODE/SDE solvers, constant step size, equally spaced output times
- `RecursiveCheckpointAdjoint` (discretise-then-optimise) and forward-mode AD
- Along-trajectory metrics: separation distance, normalised separation distance, Liu Index
- Geographic unit conversions (metres ↔ degrees)
- xarray (zarr/netCDF) dataset loading into JAX pytrees

**Not in scope (deferred):**
- Adaptive step size
- Non-rectilinear or curvilinear grids
- Higher-order interpolation (cubic, etc.)
- Arakawa C/D grids
- Distributed simulation

---

## 1. Resolved Design Decisions

### 1.1 State and axis conventions

- State vector `y` has shape `(2,)` = `[lat, lon]` in **degrees** (geographic coordinates).
- Forcing arrays have axis order `(time, lat, lon)`. Coordinate arrays are 1-D JAX float arrays.
- Time is represented as **seconds since epoch** (float scalar, compatible with `datetime64[s]` → int → float cast).
- The canonical dtype is `jnp.float32` unless the user overrides at load time.

### 1.2 Term (dynamics) API

User-supplied dynamics callables must have the signature:

```python
def my_term(t: Float[Array, ""], y: Float[Array, "2"], args: PyTree) -> Float[Array, "2"]:
    ...  # returns velocity in degrees/second
```

- `t`: scalar time in seconds.
- `y`: state `[lat, lon]` in degrees.
- `args`: arbitrary JAX-compatible pytree (e.g. a `Dataset` instance).
- Return: velocity `[dlat/dt, dlon/dt]` in **degrees per second**.

The solver multiplies the return value by `dt` to form the state increment. The user is responsible for unit conversion (utility functions are provided).

### 1.3 SDE API

SDE integration uses two separate callables:

```python
def drift(t, y, args) -> Float[Array, "2"]:       # deterministic velocity, deg/s
def diffusion(t, y, args) -> Float[Array, "2 n"]:  # diffusion matrix, deg/s^0.5
```

The noise increment at each step is `dW ~ N(0, sqrt(dt)) ∈ R^n`. For standard isotropic 2D diffusion, `n=2`. The user controls `n` via the shape of the diffusion return value. The SDE increment is:

```
y_{k+1} = y_k + drift(t_k, y_k, args) * dt + diffusion(t_k, y_k, args) @ dW_k
```

For Heun (Milstein-style), the SDE step uses the Heun/Stratonovich correction. The noise is **not** restricted to Brownian motion; the user may supply any arbitrary diffusion matrix callable.

All `n_samples` realisations are vectorised via `jax.vmap` over pre-split PRNG keys. Noise is pre-sampled as a `(n_steps, n)` array before the scan loop, enabling clean `lax.scan` over both noise and time simultaneously.

### 1.4 Adjoint and differentiation modes

Two `adjoint` modes:

| Mode | Mechanism | Notes |
|---|---|---|
| `"recursive_checkpoint"` | `lax.scan` over steps; JAX's built-in scan VJP differentiates automatically through the loop. For memory-efficient O(log N) checkpointing, wrap scan body with `jax.checkpoint`. | Default. "Discretise-then-optimise." |
| `"forward"` | User calls `jax.jvp(solve_ode, ...)` directly. The scan still runs; `jax.jvp` handles forward-mode tangent propagation. | No special code needed inside the solver. |

Both modes support `jax.jit`. The solver never uses Python-level `for` loops or `lax.while_loop` without a bounded guard, ensuring gradient correctness.

### 1.5 Interpolation

Bilinear interpolation on a rectilinear A-grid is implemented natively without third-party interpolation libraries. The algorithm:

1. **Floor index lookup** using `jnp.searchsorted` (returns static-shape result, safe for jit).
2. **Clamp** indices to `[0, N-2]` with `jnp.clip`.
3. **Compute linear weights** in each axis: `w = (x - x[i]) / (x[i+1] - x[i])`.
4. **Bilinear blend** of the 4 surrounding grid values (no branching, fully differentiable).
5. **Temporal interpolation** applied first (1D linear), then spatial bilinear.

No `jnp.where` branching on index values is needed — the index clamping handles boundaries. `searchsorted` produces concrete integer indices (not traced), so no `jax.lax.dynamic_slice` is required provided input coordinates are static. If coordinates are traced, use `jax.lax.dynamic_slice` with the clamped index.

**Longitude wrap**: handled by mapping longitude to `[lon_min, lon_min + 360)` before lookup (modular arithmetic, no branching).

### 1.6 Geographic conversions and constants

```python
EARTH_RADIUS = 6_371_008.8  # metres  (matches pastax)

def meters_to_degrees(arr: Float[Array, "... 2"], lat: Float) -> Float[Array, "... 2"]:
    # arr[..., 0] = north-south metres → degrees (arc length)
    # arr[..., 1] = east-west metres → degrees (arc length / cos(lat))
    rad = arr / EARTH_RADIUS          # radians
    deg = jnp.degrees(rad)
    lon_scale = jnp.cos(jnp.radians(lat))
    return deg.at[..., 1].divide(lon_scale)

def degrees_to_meters(arr: Float[Array, "... 2"], lat: Float) -> Float[Array, "... 2"]:
    # inverse of meters_to_degrees
```

Safe-for-grad operations:

```python
def safe_sqrt(x):
    mask = x > 0.0
    return jnp.where(mask, jnp.sqrt(jnp.where(mask, x, 1.0)), 0.0)

def safe_log(x):
    mask = x > 0.0
    return jnp.where(mask, jnp.log(jnp.where(mask, x, 1.0)), -jnp.inf)

def safe_divide(a, b):
    mask = b != 0.0
    return jnp.where(mask, jnp.where(mask, a / jnp.where(mask, b, 1.0), 0.0), 0.0)
```

---

## 2. Package Layout

```
src/
  molisanax/
    __init__.py          # public API re-exports
    _types.py            # jaxtyping aliases (Float, Int, Array)
    geo.py               # EARTH_RADIUS, safe_sqrt/log/divide, haversine,
                         # meters_to_degrees, degrees_to_meters
    interpolation.py     # bilinear_interp_2d, trilinear_interp (space+time)
    forcing.py           # Field(eqx.Module), Dataset(eqx.Module)
    solver.py            # AbstractSolver, Euler, Heun, solve_ode, solve_sde
    metrics.py           # separation_distance, normalized_separation_distance, liu_index

tests/
  conftest.py
  test_geo.py
  test_interpolation.py
  test_forcing.py
  test_solver.py
  test_metrics.py

docs/
  project_plan.md
  implementation_plan.md
  project_status.md

pyproject.toml
README.md
```

---

## 3. Module Specifications

### 3.1 `geo.py`

```python
EARTH_RADIUS: float = 6_371_008.8  # m

safe_sqrt(x) -> Array
safe_log(x) -> Array
safe_divide(a, b) -> Array
haversine(y1: Float[Array, "2"], y2: Float[Array, "2"]) -> Float[Array, ""]
    # returns great-circle distance in metres
meters_to_degrees(arr: Float[Array, "... 2"], lat: Float) -> Float[Array, "... 2"]
degrees_to_meters(arr: Float[Array, "... 2"], lat: Float) -> Float[Array, "... 2"]
```

### 3.2 `interpolation.py`

```python
def linear_interp_1d(
    values: Float[Array, "n"],
    coords: Float[Array, "n"],
    x: Float[Array, ""],
) -> Float[Array, ""]:
    # 1D linear interpolation (for time)

def bilinear_interp_2d(
    values: Float[Array, "lat lon"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
) -> Float[Array, ""]:
    # Bilinear interpolation on a 2D rectilinear grid

def spatiotemporal_interp(
    values: Float[Array, "time lat lon"],
    t_coords: Float[Array, "time"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    t: Float[Array, ""],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
) -> Float[Array, ""]:
    # Trilinear: first interpolate in time, then bilinear in space
    # Alternatively: bilinear in space at t_lo and t_hi, then linear blend in time
```

### 3.3 `forcing.py`

```python
class Field(eqx.Module):
    values: Float[Array, "time lat lon"]
    t_coords: Float[Array, "time"]
    lat_coords: Float[Array, "lat"]
    lon_coords: Float[Array, "lon"]

    def interp(
        self,
        t: Float[Array, ""],
        lat: Float[Array, ""],
        lon: Float[Array, ""],
    ) -> Float[Array, ""]:
        return spatiotemporal_interp(
            self.values, self.t_coords, self.lat_coords, self.lon_coords,
            t, lat, lon,
        )

class Dataset(eqx.Module):
    fields: dict[str, Field]          # equinox allows dict fields

    @staticmethod
    def from_xarray(
        ds: xr.Dataset,
        fields: dict[str, str],       # {internal_name: xarray_var_name}
        coordinates: dict[str, str],  # {"time": "...", "lat": "...", "lon": "..."}
        dtype: DTypeLike = jnp.float32,
    ) -> "Dataset":
        # converts xarray arrays to JAX arrays
        # time: datetime64[s] → int → float (seconds since epoch)
        # returns Dataset with all fields loaded into device memory
```

### 3.4 `solver.py`

```python
class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
    ) -> Float[Array, "2"]:
        ...

class Euler(AbstractSolver):
    def step(self, term, t, y, dt, args):
        return y + term(t, y, args) * dt

class Heun(AbstractSolver):
    def step(self, term, t, y, dt, args):
        k1 = term(t, y, args)
        k2 = term(t + dt, y + k1 * dt, args)
        return y + 0.5 * (k1 + k2) * dt

def solve_ode(
    term: Callable[[Float, Float[Array, "2"], PyTree], Float[Array, "2"]],
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, "time"],
    solver: AbstractSolver = Heun(),
    *,
    adjoint: Literal["recursive_checkpoint", "forward"] = "recursive_checkpoint",
) -> Float[Array, "time 2"]:
    # Integrates ODE from ts[0] to ts[-1] with constant dt = ts[1]-ts[0]
    # Returns array of shape (len(ts), 2) including y0 at ts[0]
    # Uses lax.scan; jax.checkpoint applied to body for memory efficiency

def solve_sde(
    drift: Callable[[Float, Float[Array, "2"], PyTree], Float[Array, "2"]],
    diffusion: Callable[[Float, Float[Array, "2"], PyTree], Float[Array, "2 n"]],
    args: PyTree,
    y0: Float[Array, "2"],
    ts: Float[Array, "time"],
    key: Key[Array, ""],
    n_samples: int,
    solver: AbstractSolver = Heun(),
) -> Float[Array, "samples time 2"]:
    # vmaps solve_one over n_samples PRNG keys
    # Within each realisation, noise increments are pre-sampled and passed into lax.scan
```

**Implementation note on SDE step for Heun**: Euler-Maruyama for SDE:
```
y1 = y0 + drift(t, y0, args)*dt + diffusion(t, y0, args) @ dW
```
Heun (Milstein-style predictor-corrector for SDE):
```
y_pred = y0 + drift(t, y0, args)*dt + diffusion(t, y0, args) @ dW
y1 = y0 + 0.5*(drift(t, y0, args) + drift(t+dt, y_pred, args))*dt
       + 0.5*(diffusion(t, y0, args) + diffusion(t+dt, y_pred, args)) @ dW
```

### 3.5 `metrics.py`

```python
def separation_distance(
    y: Float[Array, "time 2"],
    y_ref: Float[Array, "time 2"],
) -> Float[Array, "time"]:
    # Haversine distance at each time step, in metres

def normalized_separation_distance(
    y: Float[Array, "time 2"],
    y_ref: Float[Array, "time 2"],
) -> Float[Array, "time"]:
    # separation_distance[t] / (cumulative trajectory length of y_ref up to t)
    # uses safe_divide to handle zero-length reference

def liu_index(
    y: Float[Array, "time 2"],
    y_ref: Float[Array, "time 2"],
) -> Float[Array, "time"]:
    # cumsum(separation_distance) / cumsum(trajectory_length_of_y_ref)
    # This matches the definition in pastax and Liu et al. 2011

# All three functions are vmap-compatible:
# apply jax.vmap to handle Float[Array, "samples time 2"] inputs → Float[Array, "samples time"]
```

---

## 4. Key Differentiability Constraints

| Location | Risk | Mitigation |
|---|---|---|
| Index lookup in interpolation | `searchsorted` returns concrete int in jit if coords are static | Use `lax.dynamic_slice` if coords are traced parameters |
| Bilinear weights | `(x - x[i]) / (x[i+1] - x[i])` can be zero denom at boundary | Clamp indices so `x[i+1] != x[i]` is guaranteed for equally-spaced grids |
| Haversine at coincident points | `sqrt(0)` has undefined gradient | `safe_sqrt` via `jnp.where(mask, x, 1.0)` pattern |
| Cumulative sum for Liu Index | `sum / sum` at `t=0` gives `0/0` | `safe_divide` |
| SDE noise in lax.scan | Cannot sample inside scan without carrying PRNG state | Pre-sample full noise array `(n_steps, n)` before scan; pass as part of scanned input |
| Python-level loop in solver | Breaks grad | Always use `lax.scan`, never Python `for` |

---

## 5. Dependencies

**Mandatory**: `jax`, `equinox`, `jaxtyping`

**Required for forcing loading**: `xarray`, `zarr`, `netcdf4` (optional install extras)

**Linting/formatting/typing**: `ruff`, `pyright`

**No diffrax, interpax, or other JAX ecosystem libs in core runtime.**

---

## 6. pyproject.toml sketch

```toml
[project]
name = "molisanax"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "jax>=0.4.30",
    "equinox>=0.11.0",
    "jaxtyping>=0.2.30",
]

[project.optional-dependencies]
forcing = ["xarray", "zarr", "netcdf4", "numpy"]
dev = ["pytest", "ruff", "pyright"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/molisanax"]

[tool.ruff]
line-length = 100

[tool.pyright]
pythonVersion = "3.11"
include = ["src/molisanax"]
```

---

## 7. Iteration Order

1. **Scaffold**: `pyproject.toml`, `src/molisanax/__init__.py`, `tests/conftest.py` — pip install works, `import molisanax` works.
2. **geo.py + tests**: constants, safe ops, haversine, unit conversions — all tested analytically.
3. **interpolation.py + tests**: 1D linear, 2D bilinear, spatiotemporal — test against known closed-form values.
4. **forcing.py + tests**: `Field`, `Dataset.from_xarray` — test with a synthetic in-memory xarray dataset.
5. **solver.py + tests**: `Euler`, `Heun`, `solve_ode` — test with analytical forcing (`u=const`), check convergence. Then `solve_sde` — test that ensemble mean tracks ODE solution.
6. **metrics.py + tests**: all three metrics, test scalar and vmapped (ensemble) cases.
7. **README + project_status.md**: update before commit.
8. **Commit and push**.

---

## 8. Acceptance Criteria per Iteration End

- `pip install -e .` succeeds from a clean env.
- `pytest -q` is green (0 failures, 0 errors).
- `pyright src/molisanax` reports 0 errors.
- `jax.jit(solve_ode)(...)` compiles without ConcretizationTypeError.
- `jax.grad(lambda y0: solve_ode(..., y0=y0).sum())(y0_init)` returns a finite array.
- `jax.jvp(solve_ode, ...)` returns finite tangents.
- Uniform-field ODE test: `u_lat=u_lon=1e-4 deg/s` over `T` seconds → final position matches `y0 + u*T` to within `1e-5 deg`.
