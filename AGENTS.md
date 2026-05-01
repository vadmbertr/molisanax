# molisanax Project Context

## Purpose
The purpose of the project is to implement in JAX a differentiable lagrangian simulator for ocean surface trajectories.

## Contracts
- **Exposes**: library API under `src/molisanax`, documentation under `docs`, tests under `tests`.
- **Guarantees**: behavior changes are test-backed; docs track current intent and constraints
- **Expects**: contributors optimize for correctness, backward auto-differentition, and forward and backard performances.

## Dependencies
- **Uses**: 
JAX and Equinox are mandatory dependencies; optional dependencies (if deemed useful) from the Equinox ecosystem are `diffrax`, `quax`, `unxt`, and `lineax`.
Should use a linter and a formatter, and `pyright` as a static type checker.
- **Used by**: regression tests, and coding-agent hardening exercises.

## Invariants
- `pytest -q` must stay green after behavior changes.
- Documentation must explicitly reflect mission or contract shifts.

## Key Decisions
- Treat `docs/project_plan.md` as historical baseline plus dated reassessments.
- Use `AGENTS.md` as canonical project context and `CLAUDE.md` for orchestration instructions.

## Commands
- `pytest -q` - run full test suite.

## Project Structure
- `src/molisanax/` - package runtime.
- `tests/` - regression tests.
- `docs/` - project plans and licensing guidance.

## Behavior Requirements For Agents
- Prefer test-first or test-coupled changes for any behavioral modification.
- Update context docs when project goals, contracts, or boundaries change.
- Report concrete verification evidence (commands and outcomes), not assumptions.

## Boundaries
- Safe to edit: `src/`, `tests/`, `docs/`, `README.md`.
- Do not copy source text from `nemo_abl1d_GMD_2020` into MIT-licensed runtime code.
