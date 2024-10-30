import jax
from jax import numpy as jnp


def rpsnorm(ps: jax.Array, wxyz: float) -> jax.Array:
    return wxyz * jnp.sum(jnp.real(jnp.conjugate(ps) * ps))


def overlap(pl: jax.Array, pr: jax.Array, wxyz: float) -> jax.Array:
    return wxyz * jnp.sum(jnp.real(jnp.conjugate(pl) * pr))
