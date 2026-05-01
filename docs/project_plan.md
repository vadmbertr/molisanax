## GOAL
Your goal is to implement in JAX/Equinox a differentiable lagrangian simulator for ocean surface trajectories.

## RULES
Simulating a trajectory, or an ensemble of trajectories should be fully differentiable via JAX auto-differentiation.

The code should allow to simulate a trajectory by solving an ODE, or an ensemble of trajectory by solving an SDE (in this case, the residual noise can take a variety of different form, we do not restrict to use a Brownian residual).
The code should allow an user to use an arbitrary term (as a function, assumed to be JAX jit/grad compatible, with a prescibed signature) to evolve the state of the system. This term will return a `jax.Array` that, when multiplied by the integration time step forms the RHS part of the differential equation.
The code should provide utility functions to cleanly and precisely converts a physical quantity from meters to geographic degrees, and vice-versa.
The code should allow to easily load forcings from xarray datasets (zarr or netcdf).
The code should allow to compute along trajectory metrics (with respect to a reference trajectory): separation distance, normalized separation distance, and Liu Index. Those should be appliable to ensemble of trajectories (and return an ensemble of metrics).
The code should allow to use backward-mode AD using a "RecursiveCheckpointAdjoint" (sometimes known as "discretise-then-optimise", or described as "backpropagation through the solver"), or simply use forward-mode AD.
The code base should use when necessary "safe for grads" operations (for sqrt, log, divide at 0 for example).
The code base should be divided into meaningful classes.

As it is a 1st implementation, the code should only handle for now:
- forcing fields that are equally spaced (in space and time) rectilinear Arakawa A-grids defined on geographic coordinates axis (latitude and longitude),
- (bi)linear interpolation of the forcing fields,
- simple solvers: Euler and Heun,
- constant integration step size,
- equally spaced output times.

The code should focus and put effort into producing a very efficient simulator in the forward and backward passes and a clean API to the user, rather than speculating on futur new futures that would deteriorate performances and readability.

Inspiration regarding the lagrangian simulator can be taken from:
- https://github.com/Parcels-code/parcels
- https://github.com/JuliaClimate/Drifters.jl

Inspiration regarding the integration can be taken from:
- https://docs.kidger.site/diffrax/
- https://docs.kidger.site/lineax/
- https://github.com/sciml/differentialequations.jl
- https://github.com/neuralgcm/dinosaur

Inspiration regarding the handling of the grids/forcings can be taken from:
- https://github.com/neuralgcm/coordax
- https://github.com/google/jax-datetime
- https://github.com/earth-mover/Zarrs.jl
- https://github.com/GalacticDynamics/unxt
- https://github.com/earth-mover/icechunk
- https://github.com/neuralgcm/dinosaur
- https://github.com/google-deepmind/xarray_jax
- https://github.com/juliaclimate/MeshArrays.jl

Inspiration regarding the potential creation of Pytrees with pre-attached basic mathematical operations:
- https://github.com/nstarman/quax
- https://docs.kidger.site/lineax/

Bare in mind that those code bases come with a licence that should be respected. In doubt, never copy code from existing code bases.

A (messy) code base that attempted the same goal as yours can be find at github.com/vadmbertr/pastax. You can safely reuse code from this one. 

## STEPS

- read/scan carefully the different inspiration sources (using different subagents)
- prepare and write a plan in  `docs/implementation_plan.md` that should notably make clear
  - how interpolation is performed,
  - how integration is implemented,
  - how randomness is handled (for SDE),
  - where/when the mandatory differentiability functionality is a big constraint (it can always be aleviate) 
- proceed with the plan with a ralph loop. at the end of each iteration, the code should be pip installable, and fonctional, with tests corresponding to the current level of implementation. all code should be commited locally and pushed to the GH repos. large development should be done in dedicated branches and merged only when fonctional
- the README.md should be in phase with the code before each commit (content, desription, CLI, algorithm, project status, ...)
- at the end of each iteration, keep a clear and short description of the project status in `docs/project_status.md`
