import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass


@partial(register_dataclass,
         data_fields=['rho', 'chi', 'tau', 'current', 'sdens', 'sodens'],
         meta_fields=[])
@dataclass
class Densities:
    rho: jax.Array
    chi: jax.Array
    tau: jax.Array
    current: jax.Array
    sdens: jax.Array
    sodens: jax.Array


def init_densities(grids) -> Densities:
    shape4d = (grids.nx, grids.ny, grids.nz, 2)
    shape5d = (grids.nx, grids.ny, grids.nz, 3, 2)

    default_kwargs = {
        'rho': jnp.zeros(shape4d, dtype=jnp.float64),
        'chi': jnp.zeros(shape4d, dtype=jnp.float64),
        'tau': jnp.zeros(shape4d, dtype=jnp.float64),
        'current': jnp.zeros(shape5d, dtype=jnp.float64),
        'sdens': jnp.zeros(shape5d, dtype=jnp.float64),
        'sodens': jnp.zeros(shape5d, dtype=jnp.float64),
    }

    return Densities(**default_kwargs)
