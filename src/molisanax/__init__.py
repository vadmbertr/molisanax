"""molisanax: Differentiable Lagrangian simulator for ocean surface trajectories."""

from .forcing import Dataset, Field
from .geo import (
    EARTH_RADIUS,
    degrees_to_meters,
    haversine,
    meters_to_degrees,
    safe_divide,
    safe_log,
    safe_sqrt,
)
from .interpolation import bilinear_interp_2d, linear_interp_1d, spatiotemporal_interp
from .metrics import liu_index, normalized_separation_distance, separation_distance
from .solver import Euler, Heun, solve_ode, solve_sde

__version__ = "0.1.0"

__all__ = [
    # geo
    "EARTH_RADIUS",
    "safe_sqrt",
    "safe_log",
    "safe_divide",
    "haversine",
    "meters_to_degrees",
    "degrees_to_meters",
    # interpolation
    "linear_interp_1d",
    "bilinear_interp_2d",
    "spatiotemporal_interp",
    # forcing
    "Field",
    "Dataset",
    # solver
    "Euler",
    "Heun",
    "solve_ode",
    "solve_sde",
    # metrics
    "separation_distance",
    "normalized_separation_distance",
    "liu_index",
]
