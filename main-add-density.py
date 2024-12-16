import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities, add_density_jit
from meanfield import init_meanfield
from levels import init_levels
from static import init_static
from coulomb import init_coulomb, poisson
from test import *

jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_platform_name', 'cpu')

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

levels.wocc = levels.wocc.at[...].set(1.0)
levels.wstates = levels.wstates.at[...].set(1.0)
levels.wguv = levels.wguv.at[...].set(0.0)
levels.pairwg = levels.pairwg.at[...].set(1.0)


levels.psi = levels.psi.at[...].set(load5d('psi'))
densities.rho = densities.rho.at[...].set(load4d_real('rho'))
densities.chi = densities.chi.at[...].set(load4d_real('chi'))
densities.tau = densities.tau.at[...].set(load4d_real('tau'))
densities.current = densities.current.at[...].set(load5d_real('current'))
densities.sdens = densities.sdens.at[...].set(load5d_real('sdens'))
densities.sodens = densities.sodens.at[...].set(load5d_real('sodens'))

rho = load4d_real('res_rho')
chi = load4d_real('res_chi')
tau = load4d_real('res_tau')
current = load5d_real('res_current')
sdens = load5d_real('res_sdens')
sodens = load5d_real('res_sodens')


res = add_density_jit(densities, grids, levels)

print(jnp.max(jnp.abs(rho - res.rho)))
print(jnp.max(jnp.abs(chi - res.chi)))
print(jnp.max(jnp.abs(tau - res.tau)))
print(jnp.max(jnp.abs(current - res.current)))
print(jnp.max(jnp.abs(sdens - res.sdens)))
print(jnp.max(jnp.abs(sodens - res.sodens)))

#8.326672684688674e-17
#0.0
#1.6653345369377348e-16
#1.734723475976807e-17
#8.131516293641283e-18
#1.1275702593849246e-17

import timeit
start_time = timeit.default_timer()
for _ in range(2000):
    res = add_density_jit(densities, grids, levels)
    jax.block_until_ready(res)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 2000

print(average_time)
# 0.0625475561041385
