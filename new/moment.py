import jax
from dataclasses import dataclass, field
from functools import partial

@partial(jax.tree_util.register_dataclass,
data_fields=['cmtot'],
meta_fields=[])
@dataclass
class Moment:
    cmtot: jax.Array


def init_moment(cmtot):
    return Moment(cmtot)
