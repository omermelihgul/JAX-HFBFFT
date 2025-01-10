import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, diagstep_jit, diagstep
from coulomb import init_coulomb, poisson
from energies import init_energies
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
energies = init_energies()
params.iteration = 4

levels.wocc = levels.wocc.at[...].set(1.0)
levels.wstates = levels.wstates.at[...].set(1.0)
levels.wguv = levels.wguv.at[...].set(0.0)
levels.pairwg = levels.pairwg.at[...].set(1.0)
forces.tbcs = True

# levels.psi = levels.psi.at[...].set(load5d('psi'))
# levels.hampsi = levels.hampsi.at[...].set(load5d('res_hampsi'))
# levels.hmfpsi = levels.hampsi.at[...].set(load5d('res_hmfpsi'))

levels.psi = levels.psi.at[...].set(1)
levels.hampsi = levels.hampsi.at[...].set(0.4)
levels.hmfpsi = levels.hampsi.at[...].set(1.2)

sp_energy = load1d_real('sp_energy', 132)

energies, levels, static = diagstep_jit(energies, forces, grids, levels, static, True, True)


print(jnp.max(jnp.abs(sp_energy - levels.sp_energy)))
# 4.902744876744691e-13



import timeit
start_time = timeit.default_timer()
for _ in range(1000):
    energies, levels, static = diagstep_jit(energies, forces, grids, levels, static, True, True)
    jax.block_until_ready(energies)
    jax.block_until_ready(static)
    jax.block_until_ready(levels)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 1000

print(average_time)
# 0.030063006047159432










# levels.hampsi = levels.hampsi.at[...].set(load5d('hampsi'))
# levels.lagrange = levels.lagrange.at[...].set(load5d('lagrange'))
# levels.hmfpsi = levels.hmfpsi.at[...].set(load5d('hmfpsi'))
# levels.delpsi = levels.delpsi.at[...].set(load5d('delpsi'))


# psi = load5d('res_psi')
# hampsi = load5d('res_hampsi')
# lagrange = load5d('res_lagrange')
# psi_temp = load5d('res_psi_temp')
# hmfpsi = load5d('res_hmfpsi')
# delpsi = load5d('res_delpsi')

# efluct1 = load1d_real('efluct1', 1)
# efluct2 = load1d_real('efluct2', 1)
# efluct1q = load1d_real('efluct1q', 2)
# efluct2q = load1d_real('efluct2q', 2)

# sp_energy = load1d_real('sp_energy', 132)
# sp_norm = load1d_real('sp_norm', 132)

# hmatrix = load3d('hmatrix', 82, 82, 2)
# gapmatrix = load3d('gapmatrix', 82, 82, 2)
# symcond = load3d('symcond', 82, 82, 2)

# forces.tbcs = True

# # static, levels, energies
# res1, res2, res3 = diagstep_jit(energies, forces, grids, levels, static, True, True)
# print(jnp.max(levels.psi))


# print(jnp.max(psi))
# print(forces.tbcs)
