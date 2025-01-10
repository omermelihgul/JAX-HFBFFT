import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities, add_density_jit
from meanfield import init_meanfield, hpsi01
from levels import init_levels
from static import init_static, grstep_vmap
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

forces.tbcs = True

levels.psi = levels.psi.at[...].set(load5d('psi'))
levels.lagrange = levels.psi.at[...].set(load5d('lagrange'))

levels.sp_energy = levels.sp_energy.at[...].set(load1d_real('sp_energy', 132))
levels.sp_efluct1 = levels.sp_efluct1.at[...].set(load1d_real('sp_efluct1', 132))

denerg = load1d_real('denerg', 1)

meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
meanfield.divaq = meanfield.divaq.at[...].set(load4d_real('divaq'))
meanfield.v_pair = meanfield.v_pair.at[...].set(load4d_real('v_pair'))
meanfield.aq = meanfield.aq.at[...].set(load5d_real('aq'))
meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
meanfield.dbmass = meanfield.dbmass.at[...].set(load5d_real('dbmass'))

#psin = load4d('psin')
#res_psin = load4d('res_psin')
#res_psi_mf = load4d('res_psi_mf')
#delpsi = load4d('delpsi')

#res1, res2, res3 = hpsi01(grids, meanfield, 0, 1.0, 0.0, psin)


#print(jnp.max(jnp.abs(res_psin - res1)))
#print(jnp.max(jnp.abs(res_psi_mf - res2)))
#print(jnp.max(jnp.abs(delpsi - res3)))
#print(jnp.max(res_psin))
#print(jnp.max(res1))
res_psi = load5d('res_psi')
res_delpsi = load5d('res_delpsi')
res_hmfpsi = load5d('res_hmfpsi')
res_hampsi = load5d('res_hampsi')

size = 132
nst = jnp.arange(0, 132)
spe_mf = jnp.arange(0, 132)
iq = jnp.zeros(size, dtype=int)
iq = iq.at[size // 2:].set(1)
static.x0dmp = 1.0908270059034080
print(forces.h2ma)
forces.h2ma = 20.735530853271484
psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi = grstep_vmap(params, forces, grids, meanfield, levels, static, nst, levels.isospin, levels.sp_energy, levels.psi, levels.lagrange)

params.iteration = 4

print(jnp.sum(res_psi))
print(jnp.sum(psin))
print(jnp.max(jnp.abs(psin - res_psi)))

print(jnp.sum(res_hmfpsi))
print(jnp.sum(hmfpsi))
print(jnp.max(jnp.abs(hmfpsi - res_hmfpsi)))

print(jnp.sum(res_delpsi))
print(jnp.sum(delpsi))
print(jnp.max(jnp.abs(res_delpsi - delpsi)))
print(jnp.max(jnp.abs(res_hampsi - psi_mf)))

print(jnp.sum(denerg))

import timeit
start_time = timeit.default_timer()
for i in range(5):
    psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi = grstep_vmap(params, forces, grids, meanfield, levels, static, nst, levels.isospin, levels.sp_energy, levels.psi, levels.lagrange)
    jax.block_until_ready(psin)
    jax.block_until_ready(psi_mf)
    jax.block_until_ready(spe_mf_new)
    jax.block_until_ready(denerg)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 5

print(average_time)
#

'''
def laplace(grids, forces, wg, wguv, psin, v_pairmax, e0inv):
    weight = jnp.maximum(wg, 0.1)
    weightuv = jnp.maximum(wguv, 0.1)

    kfacx = (jnp.pi + jnp.pi) / (grids.dx * grids.nx)
    kfacy = (jnp.pi + jnp.pi) / (grids.dy * grids.ny)
    kfacz = (jnp.pi + jnp.pi) / (grids.dz * grids.nz)

    indices_x = jnp.concatenate([
        jnp.arange(grids.nx // 2),
        jnp.arange(grids.nx // 2, 0, -1)
    ])
    indices_y = jnp.concatenate([
        jnp.arange(grids.ny // 2),
        jnp.arange(grids.ny // 2, 0, -1)
    ])
    indices_z = jnp.concatenate([
        jnp.arange(grids.nz // 2),
        jnp.arange(grids.nz // 2, 0, -1)
    ])

    k2facx = -((indices_x * kfacx) ** 2)[:,jnp.newaxis,jnp.newaxis]
    k2facy = -((indices_y * kfacy) ** 2)[jnp.newaxis,:,jnp.newaxis]
    k2facz = -((indices_z * kfacz) ** 2)[jnp.newaxis,jnp.newaxis,:]
    psout = jnp.fft.fftn(psin, axes=(1, 2, 3), norm='forward')

    denominator = (
        ((weight * (e0inv - forces.h2ma * (k2facx + k2facy + k2facz))) + 0.5 * weightuv * v_pairmax) )

    psout = psout.at[...].divide(
        denominator[jnp.newaxis,...]
    )

    psout = psout.at[...].set(
        jnp.fft.ifftn(psout, axes=(1, 2, 3), norm='forward')
    )

    return psout


psout = load4d('psout')
res_psout = load4d('res_psout')

res1 = laplace(grids, forces, 1.0, 0.0, psout, 0, 100)

#print(jnp.max(ps1))
#print(ps2[0,31,:,12])
#print(res[0,31,:,12])
forces.h2ma = 20.735530853271484
print(forces.h2ma)
print(res_psout[0,34,:10,1])
print(res1[0,34,:10,1])
print(jnp.max(jnp.abs(res_psout - res1)))

'''
