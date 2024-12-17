import jax
from jax import numpy as jnp
from dataclasses import dataclass


@jax.tree_util.register_dataclass
@dataclass
class Energies:
    efluct1: jax.Array
    efluct2: jax.Array
    efluct1q: jax.Array
    efluct2q: jax.Array


def init_energies(**kwargs) -> Energies:
    default_kwargs = {
        'efluct1': jnp.zeros(1, dtype=jnp.float64),
        'efluct2': jnp.zeros(1, dtype=jnp.float64),
        'efluct1q': jnp.zeros(2, dtype=jnp.float64),
        'efluct2q': jnp.zeros(2, dtype=jnp.float64),
    }

    default_kwargs.update(kwargs)

    return Energies(**default_kwargs)
