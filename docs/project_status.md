# Project Status

*Last updated: 2026-05-01*

## Version: 0.1.0 — Second iteration (API refinement)

### What is implemented

**Core library** (`src/molisanax/`):

| Module | Status | Notes |
|---|---|---|
| `_safe_math.py` | Done | `safe_sqrt`, `safe_log`, `safe_divide` — gradient-safe math utilities |
| `geo.py` | Done | `EARTH_RADIUS=6_371_008.8 m`, `haversine`, `meters_to_degrees`, `degrees_to_meters` |
| `interpolation.py` | Done | 1D linear, 2D bilinear, trilinear (time+space); native JAX, no interpax |
| `forcing.py` | Done | `Field` (with `.interp` and `.neighborhood`), `Dataset.from_xarray`; (time, lat, lon) axis order |
| `solver.py` | Done | `Euler`, `Heun`; unified `solve()` with auto ODE/SDE detection via `jax.eval_shape` |
| `metrics.py` | Done | `separation_distance`, `normalized_separation_distance`, `liu_index`; all support `ensemble=True` |

**Tests**: 73 tests, all passing (`pytest -q`).

**Differentiability**: `jax.grad` and `jax.jvp` through `solve()` in ODE mode verified in tests. `jax.vmap` over SDE ensemble verified.

### API highlights

**Unified solver**: a single `solve(term, args, y0, ts, solver, *, key, n_samples)` function detects ODE vs SDE from the term's return type at call time (via `jax.eval_shape`):
- ODE: `term(t, y, args) -> Float[Array, "2"]` → no key required, returns `(T, 2)`
- SDE: `term(t, y, args) -> tuple[Float[Array, "2"], Float[Array, "2"]]` → requires `key`; returns `(T, 2)` for a single realisation or `(S, T, 2)` for an ensemble

**SDE formulation**: `dy = (f + g * z) * dt` where `z ~ N(0, I₂)` is drawn fresh each step by the solver. `g` has units of deg/s — it is a noise-amplitude velocity, not a diffusion matrix. Noise is pre-sampled before the `lax.scan` loop.

**Neighbourhood extraction**: `Field.neighborhood(t, lat, lon, t_window, lat_window, lon_window)` returns a window of raw grid values via `lax.dynamic_slice` (jit-compatible, grad-compatible w.r.t. the query point).

**Ensemble metrics**: all metric functions accept `ensemble=True` to automatically vmap over the first (sample) axis.

### What is not yet implemented (deferred)

- Adaptive step size
- Higher-order interpolation (cubic, splines)
- Non-rectilinear or curvilinear grids (e.g. Arakawa C/D)
- Longitude wrap-around (periodic boundary) in interpolation
- CLI or notebook examples
- Benchmark suite (convergence order, performance)

### Known limitations

- **Longitude wrap**: `bilinear_interp_2d` does not handle the 360°→0° discontinuity. Use `[-180, 180]` coordinates without wrap for now.
- **Extrapolation**: Interpolation clamps to grid boundary outside the domain.
- **Memory for long trajectories**: reverse-mode AD through `solve()` stores one checkpoint per `lax.scan` step (O(T) memory). Use forward-mode (`jax.jvp`) for very long trajectories.

### Architecture summary

- No diffrax/interpax runtime dependency.
- Solvers use `jax.lax.scan` + `jax.checkpoint` for memory-efficient reverse-mode AD.
- State: `Float[Array, "2"]` = `[lat, lon]` in degrees.
- Term signature unified across ODE and SDE; mode detected at Python level, not at trace time.
