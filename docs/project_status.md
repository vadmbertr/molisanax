# Project Status

*Last updated: 2026-05-01*

## Version: 0.1.0 — First Iteration

### What is implemented

**Core library** (`src/molisanax/`):

| Module | Status | Notes |
|---|---|---|
| `geo.py` | Done | EARTH_RADIUS=6_371_008.8 m, safe_sqrt/log/divide, haversine, meters↔degrees |
| `interpolation.py` | Done | 1D linear, 2D bilinear, trilinear (time+space); native JAX, no interpax |
| `forcing.py` | Done | `Field`, `Dataset.from_xarray`; (time, lat, lon) axis order |
| `solver.py` | Done | `Euler`, `Heun`; `solve_ode` and `solve_sde` with `lax.scan` loop |
| `metrics.py` | Done | `separation_distance`, `normalized_separation_distance`, `liu_index` |

**Tests**: 58 tests, all passing (`pytest -q`).

**Differentiability**: `jax.grad` and `jax.jvp` through `solve_ode` verified in tests. `jax.vmap` over SDE ensemble verified.

### What is not yet implemented (deferred)

- Adaptive step size
- Higher-order interpolation (cubic, splines)
- Non-rectilinear or curvilinear grids (e.g. Arakawa C/D)
- Longitude wrap-around (periodic boundary) in interpolation
- CLI or notebook examples
- Benchmark suite (convergence order, performance)

### Known limitations

- **Longitude wrap**: `bilinear_interp_2d` does not handle the 360°→0° longitude wrap. Forcing fields should be provided without the wrap discontinuity (e.g. `[-180, 180]` range without periodicity) for now.
- **Extrapolation**: Interpolation clamps to boundary when querying outside the forcing grid domain (equivalent to Neumann boundary condition on the velocity field).
- **Memory**: Reverse-mode AD through `solve_ode` stores O(T) checkpoints (one per `lax.scan` step). For very long trajectories, consider chunking or using forward-mode AD.
- **SDE noise**: Only Gaussian (Brownian) noise increments are pre-sampled. Arbitrary noise distributions can be implemented by the user by replacing the `jr.normal` call in a custom wrapper around `solve_sde`.

### Architecture summary

- No diffrax/interpax dependency. Solvers use `jax.lax.scan` + `jax.checkpoint`.
- State: `Float[Array, "2"]` = `[lat, lon]` in degrees.
- Term signature: `f(t, y, args) -> velocity_deg_per_s`.
- SDE uses separate drift + diffusion callables; noise pre-sampled before scan loop.
