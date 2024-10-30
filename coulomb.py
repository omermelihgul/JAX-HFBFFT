import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass

@partial(register_dataclass,
         data_fields=[],
         meta_fields=[])
@dataclass
class Coulomb:
    pass