"""Gradient-safe versions of mathematically unstable operations."""

import jax.numpy as jnp

from ._types import Array, Float

__all__ = ["safe_sqrt", "safe_log", "safe_divide"]


def safe_sqrt(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gradient-safe sqrt: returns 0 where x <= 0, avoids NaN gradients at 0."""
    mask = x > 0.0
    return jnp.where(mask, jnp.sqrt(jnp.where(mask, x, 1.0)), 0.0)


def safe_log(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gradient-safe log: returns -inf where x <= 0, avoids NaN gradients at 0."""
    mask = x > 0.0
    return jnp.where(mask, jnp.log(jnp.where(mask, x, 1.0)), -jnp.inf)


def safe_divide(
    a: Float[Array, "..."], b: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Gradient-safe divide: returns 0 where b == 0."""
    mask = b != 0.0
    return jnp.where(mask, a / jnp.where(mask, b, 1.0), 0.0)
