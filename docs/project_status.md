# Project Status

*Last updated: 2026-05-13*

## Version: 0.1.0 — Fifth iteration (Arakawa C-grid metadata layer)

### What is implemented

**Core library** (`src/molisanax/`):

| Module | Status | Notes |
|---|---|---|
| `_safe_math.py` | Done | `safe_sqrt`, `safe_log`, `safe_divide` — gradient-safe math utilities |
| `geo.py` | Done | `EARTH_RADIUS=6_371_008.8 m`, `haversine`, `meters_to_degrees`, `degrees_to_meters` |
| `interpolation.py` | Done | 1D linear, 2D bilinear, trilinear (time+space); O(1) index via closed-form floor — no searchsorted; opt-in periodic longitude via `lon_period` |
| `forcing.py` | Done | `Field` (with `.interp`, `.neighborhood`, and `stagger` metadata), `Dataset.from_xarray`/`from_arrays` (A-grid), `Dataset.from_xarray_cgrid`/`from_arrays_cgrid` (NEMO-convention C-grid); (time, lat, lon) axis order; periodic longitude via `lon_period`; both A-grid loaders accept NumPy `datetime64` time arrays and auto-convert to int seconds since the Unix epoch |
| `grid.py` | Done | `Grid` metadata: centre coordinates plus `grid_type` (`"rectilinear"`/`"curvilinear"`) and `stagger_type` (`"A"`/`"C"`); `u_face_coords()` / `v_face_coords()` derive NEMO half-cell-shifted coordinates for C-grid loaders |
| `solver.py` | Done | `Euler`, `Heun`, `RK4`; unified `solve()` — ODE/SDE detected by caller (`key`/`noise`/`n_noise`); SDE term receives `z` directly; backwards-in-time integration supported by passing a strictly decreasing `ts` |
| `metrics.py` | Done | `separation_distance`, `normalized_separation_distance`, `liu_index`; all support `ensemble=True` |

**Tests**: 137 tests, all passing (`pytest -q`).

**Documentation site**: hosted on GitHub Pages at <https://vadmbertr.github.io/molisanax/> (MyST / Jupyter Book 2 + `sphinx-ext-mystmd` for the API reference). Includes a runnable tutorial notebook at `docs/tutorial.ipynb`. Built and deployed by `.github/workflows/docs.yml`.

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

### Grid topology and stagger

| Topology | Stagger | Interpolation | Status |
|---|---|---|---|
| Rectilinear | A-grid (all variables at cell centres) | Bilinear in space, linear in time | Done |
| Rectilinear | C-grid (NEMO: U on east faces, V on north faces) | Bilinear-on-shifted-coords (each `Field` carries its own staggered 1-D coords + a `stagger` tag) — first-order accurate, exact for linear fields | Done |
| Curvilinear (2-D coord arrays) | A or C | — | Storable in `Grid` for forward compatibility; `Field.interp` raises `NotImplementedError`. Not yet supported. |
| Rectilinear | B-grid / D-grid | — | Not supported. |

The C-grid path reduces to bilinear on the field's own coordinates because shifting an equally-spaced centre grid by half a cell preserves equal spacing — no separate interpolation kernel is needed. The Delandmeter–van Sebille (2019) divergence-aware analytical C-grid scheme is not implemented; planned as a future opt-in (`Dataset.cgrid_velocity(t, lat, lon)` method) that would not alter the per-`Field` API.

### What is not yet implemented (deferred)

- Adaptive step size
- Higher-order interpolation (cubic, splines)
- Curvilinear-grid interpolation (storable structurally, but `Field.interp` raises)
- Arakawa B-grid and D-grid layouts
- Analytical C-grid velocity scheme (Doos / CGS / van Sebille 2019)
- **Land-mask awareness** — see *Known limitations* below; planned as a separate "coastal robustness" iteration
- Auto-detection of periodic longitude (currently opt-in via `lon_period`)
- Benchmark suite (convergence order, performance)

### Known limitations

- **No land handling.** `Field.interp` is naive bilinear with no mask, NaN check, or partial-cell correction. Consequences:
  - **NaN-filled land** (CMEMS convention): a single NaN corner contaminates the bilinear weighted sum → NaN velocity → NaN trajectory. molisanax raises no error; corruption is silent.
  - **Zero-filled land** (NEMO convention, A-grid): velocity asymptotes to zero as a particle approaches the coast, and the particle stalls at the boundary — the Parcels "stuck particle" failure mode.
  - **C-grid + zero-filled land faces**: the face-normal velocity vanishes correctly at coasts and particles slide along them, which is why C-grid forcing is the safer choice for coastal regions even though molisanax has no explicit coastal logic.
- **Longitude wrap**: opt-in only. Pass `lon_period=360.0` to `bilinear_interp_2d`/`spatiotemporal_interp`, or set the `lon_period` attribute on `Field` (also accepted by `Dataset.from_arrays`/`Dataset.from_xarray`). The grid must span exactly one period (no duplicated seam cell). Without it, interpolation extrapolates linearly past the boundary. C-grid U/V faces never wrap (their coordinate arrays no longer span a full period); tracer fields on a C-grid centre grid do wrap when `lon_period` is set.
- **Extrapolation**: Interpolation clamps to grid boundary outside the domain (latitude axis, and longitude when `lon_period` is not set).
- **Memory for long trajectories**: reverse-mode AD through `solve()` stores one checkpoint per `lax.scan` step (O(T) memory). Use forward-mode (`jax.jvp`) for very long trajectories.

### Architecture summary

- No diffrax/interpax runtime dependency.
- Solvers use `jax.lax.scan` + `jax.checkpoint` for memory-efficient reverse-mode AD.
- State: `Float[Array, "2"]` = `[lat, lon]` in degrees.
- Term signature unified across ODE and SDE; mode detected at Python level, not at trace time.
