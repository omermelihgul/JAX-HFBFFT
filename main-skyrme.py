import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, statichf
from coulomb import init_coulomb, poisson
from test import *

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

config = read_yaml('_config.yml')

# initialization
params = init_params(**config.get('params', {}))
forces = init_forces(params, **config.get('force', {}))
grids = init_grids(params, **config.get('grids', {}))
densities = init_densities(grids)
meanfield = init_meanfield(grids)
levels = init_levels(grids, **config.get('levels', {}))
static = init_static(levels, **config.get('static', {}))
coulomb = init_coulomb(grids)

coulomb.wcoul = coulomb.wcoul.at[...].set(load3d_real('wcoul'))
densities.rho = densities.rho.at[...].set(load4d_real('rho'))


res = poisson(grids, params, coulomb, densities.rho)

print(jnp.max(jnp.abs(coulomb.wcoul - res)))
