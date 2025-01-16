import jax
from jax import numpy as jnp
from dataclasses import dataclass
from functools import partial

@partial(jax.tree_util.register_dataclass,
data_fields = [
    'ehf',
    'ehfprev',
    'efluct1prev',
    'efluct2prev',
    'efluct1',
    'efluct2',
    'efluct1q',
    'efluct2q'
],
meta_fields = [])
@dataclass
class Energies:
    ehf: float
    ehfprev: float
    efluct1prev: float
    efluct2prev: float
    efluct1: jax.Array
    efluct2: jax.Array
    efluct1q: jax.Array
    efluct2q: jax.Array


def init_energies(**kwargs) -> Energies:
    default_kwargs = {
        'ehf': 0.0,
        'ehfprev': 0.0,
        'efluct1prev': 0.0,
        'efluct2prev': 0.0,
        'efluct1': jnp.zeros(1, dtype=jnp.float64),
        'efluct2': jnp.zeros(1, dtype=jnp.float64),
        'efluct1q': jnp.zeros(2, dtype=jnp.float64),
        'efluct2q': jnp.zeros(2, dtype=jnp.float64),
    }

    default_kwargs.update(kwargs)

    return Energies(**default_kwargs)
