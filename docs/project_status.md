# Project Status

*Last updated: 2026-05-03*

## Version: 0.1.0 — Fourth iteration (SDE API redesign)

### What is implemented

**Core library** (`src/molisanax/`):

| Module | Status | Notes |
|---|---|---|
| `_safe_math.py` | Done | `safe_sqrt`, `safe_log`, `safe_divide` — gradient-safe math utilities |
| `geo.py` | Done | `EARTH_RADIUS=6_371_008.8 m`, `haversine`, `meters_to_degrees`, `degrees_to_meters` |
| `interpolation.py` | Done | 1D linear, 2D bilinear, trilinear (time+space); O(1) index via closed-form floor — no searchsorted; opt-in periodic longitude via `lon_period` |
| `forcing.py` | Done | `Field` (with `.interp` and `.neighborhood`), `Dataset.from_xarray`, `Dataset.from_arrays`; (time, lat, lon) axis order; periodic longitude via `lon_period` |
| `solver.py` | Done | `Euler`, `Heun`; unified `solve()` — ODE/SDE detected by caller (`key`/`noise`/`n_noise`); SDE term receives `z` directly |
| `metrics.py` | Done | `separation_distance`, `normalized_separation_distance`, `liu_index`; all support `ensemble=True` |

**Tests**: 104 tests, all passing (`pytest -q`).

**Documentation site**: hosted on GitHub Pages at <https://vadmbertr.github.io/molisanax/> (MyST / Jupyter Book 2 + `sphinx-ext-mystmd` for the API reference). Includes a runnable getting-started notebook at `docs/getting_started.ipynb`. Built and deployed by `.github/workflows/docs-myst.yml`.

**Differentiability**: `jax.grad` and `jax.jvp` through `solve()` in ODE mode verified in tests. `jax.vmap` over SDE ensemble verified.

### API highlights

**Unified solver**: a single `solve(term, args, y0, ts, solver, *, key, n_samples, n_noise, noise)` function; mode is selected by the caller, not inferred from term output:
- ODE: no `key`/`noise`/`n_noise` provided. `term(t, y, args) -> Float[Array, "2"]`.
- SDE: at least one of `key`, `noise`, or `n_noise` provided. `term(t, y, args, z) -> Float[Array, "2"]` — the term receives `z` directly and returns the full velocity. `dy = term(..., z) * dt`.

**SDE noise**: two modes:
1. Auto-sampled: pass `key` + `n_noise` (and optional `n_samples`) — draws `(n_samples, n_steps, n_noise)` before `vmap`/`scan`.
2. Pre-sampled: pass `noise` of shape `(n_steps, n_noise)` or `(S, n_steps, n_noise)` — `n_noise` inferred from `noise.shape[-1]`.

**`n_noise`**: the dimension of `z`. Decoupled from the state dimension (2); supports arbitrary generative models (MDNs, flow networks, etc.).

**Neighbourhood extraction**: `Field.neighborhood(t, lat, lon, t_window, lat_window, lon_window)` returns a window of raw grid values via `lax.dynamic_slice` (jit-compatible, grad-compatible w.r.t. the query point). When the field's `lon_period` attribute is set, the longitude window wraps modulo the period instead of clamping at the edge.

**Ensemble metrics**: all metric functions accept `ensemble=True` to automatically vmap over the first (sample) axis.

### What is not yet implemented (deferred)

- Adaptive step size
- Higher-order interpolation (cubic, splines)
- Non-rectilinear or curvilinear grids (e.g. Arakawa C/D)
- Auto-detection of periodic longitude (currently opt-in via `lon_period`)
- Benchmark suite (convergence order, performance)

### Known limitations

- **Longitude wrap**: opt-in only. Pass `lon_period=360.0` to `bilinear_interp_2d`/`spatiotemporal_interp`, or set the `lon_period` attribute on `Field` (also accepted by `Dataset.from_arrays`/`Dataset.from_xarray`). The grid must span exactly one period (no duplicated seam cell). Without it, interpolation extrapolates linearly past the boundary as before.
- **Extrapolation**: Interpolation clamps to grid boundary outside the domain (latitude axis, and longitude when `lon_period` is not set).
- **Memory for long trajectories**: reverse-mode AD through `solve()` stores one checkpoint per `lax.scan` step (O(T) memory). Use forward-mode (`jax.jvp`) for very long trajectories.

### Architecture summary

- No diffrax/interpax runtime dependency.
- Solvers use `jax.lax.scan` + `jax.checkpoint` for memory-efficient reverse-mode AD.
- State: `Float[Array, "2"]` = `[lat, lon]` in degrees.
- Term signature unified across ODE and SDE; mode detected at Python level, not at trace time.
