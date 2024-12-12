import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass

@partial(register_dataclass,
         data_fields=['wcoul'],
         meta_fields=[])
@dataclass
class Coulomb:
    wcoul: jax.Array

def init_coulomb(grids) -> Coulomb:
    default_kwargs = {
        'wcoul': jnp.zeros((grids.nx, grids.ny, grids.nz), dtype=jnp.float64)
    }

    return Coulomb(**default_kwargs)
