import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities, add_density_jit
from meanfield import init_meanfield, hpsi00, hpsi01
from levels import init_levels, laplace
from static import init_static
from coulomb import init_coulomb, poisson
from test import *
from trivial import rpsnorm, overlap

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
params.iteration = 4






# def grstep(
#     params,
#     forces,
#     grids,
#     meanfield,
#     levels,
#     static,
#     nst,
#     iq,
#     spe_mf,
#     psin,
#     lagrange
# ):
#     sp_efluct1, sp_efluct2 = 0.0, 0.0

#     # Step 1
#     ps1, psi_mf = jax.lax.cond(
#         forces.tbcs,
#         lambda _: hpsi00(grids, meanfield, iq, 1.0, 0.0, psin),
#         lambda _: hpsi00(
#             grids,
#             meanfield,
#             iq,
#             levels.wstates[nst] * levels.wocc[nst],
#             levels.wstates[nst] * levels.wguv[nst] * levels.pairwg[nst],
#             psin,
#         ),
#         operand=None
#     )

#     # Step 2
#     spe_mf_new = jnp.real(overlap(psin, psi_mf, grids.wxyz))

#     # Step 3
#     def static_e0dmp_positive():
#         h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))
#         ps1_updated = ps1.at[...].add(-(h_exp * psin))

#         ps2 = jax.lax.cond(
#             forces.tbcs,
#             lambda _: laplace(grids, forces, 1.0, 0.0, ps1_updated, 0, static.e0dmp),
#             lambda _: laplace(
#                 grids,
#                 forces,
#                 levels.wocc[nst],
#                 levels.wguv[nst] * levels.pairwg[nst],
#                 ps1_updated,
#                 jnp.max(meanfield.v_pair[iq,...]),
#                 static.e0dmp
#             ),
#             operand=None
#         )

#         psin_updated = psin.at[...].add(-(static.x0dmp * ps2))
#         return psin_updated

#     def static_e0dmp_zero():
#         return psin.at[...].set(
#             (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1
#         )

#     psin = jax.lax.cond(static.e0dmp > 0.0, static_e0dmp_positive, static_e0dmp_zero)

#     # Step 4
#     ps1, hmfpsi, delpsi = hpsi01(grids, meanfield, iq, 1.0, 0.0, psin)

#     # Step 5
#     denerg = (spe_mf - spe_mf_new) / jnp.abs(spe_mf_new)

#     return psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi


# grstep_vmap = jax.vmap(jax.jit(grstep), in_axes=(None, None, None, None, None, None, 0, 0, 0, 0, 0))
#
#
#
#
#

def grstep_helper():








    pass

grstep_helper_vmap_jit = jax.vmap(
    jax.jit(grstep_helper),
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None, None)
)

def grstep(forces, grids, levels, meanfield, params, static):
    nst = jnp.arange(levels.nstmax)
    isospin = levels.isospin
    sp_energy = levels.sp_energy


    return levels, static














# def grstep(
#     params,
#     forces,
#     grids,
#     meanfield,
#     levels,
#     static,
#     nst,
#     iq,
#     spe_mf,
#     psin,
#     lagrange
# ):
#     sp_efluct1, sp_efluct2 = 0.0, 0.0

#     # Step 1
#     ps1, psi_mf = jax.lax.cond(
#         forces.tbcs,
#         lambda _: hpsi00(grids, meanfield, iq, 1.0, 0.0, psin),
#         lambda _: hpsi00(
#             grids,
#             meanfield,
#             iq,
#             levels.wstates[nst] * levels.wocc[nst],
#             levels.wstates[nst] * levels.wguv[nst] * levels.pairwg[nst],
#             psin,
#         ),
#         operand=None
#     )

#     # Step 2
#     spe_mf_new = jnp.real(overlap(psin, psi_mf, grids.wxyz))

#     # h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))

#     def e0dmp_positive(operands):
#         psin, ps1 = operands

#         ps1, h_exp = jax.lax.cond(
#             (params.iteration > 1) & (~forces.tbcs),
#             lambda _: (
#                 ps1.at[...].add(-lagrange),
#                 jnp.real(overlap(psin, ps1, grids.wxyz))
#             ),
#             lambda _: (
#                 ps1.at[...].add(-(h_exp * psin)),
#                 jnp.real(overlap(psin, ps1, grids.wxyz))
#             ),
#             operand=None
#         )




#         psin = psin.at[...].set(
#             (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1
#         )

#         return psin, 0, 0



#     def e0dmp_negative(operands):
#         psin, ps1 = operands
#         psin = psin.at[...].set(
#             (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1
#         )

#         return psin, 0, 0





#     psin, sp_efluct1, sp_efluct2 = jax.lax.cond(
#         static.e0dmp > 0.0,
#         e0dmp_positive,
#         e0dmp_negative,
#         operand=(psin, ps1)
#     )


#     # # Step 3
#     # if static.e0dmp > 0.0:
#     #     if params.iteration > 1 and not forces.tbcs:
#     #         if params.mprint > 0:
#     #             if params.iteration % params.mprint == 0:
#     #                 h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))

#     #         ps1 = ps1.at[...].add(-lagrange)
#     #     else:
#     #         h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))
#     #         ps1 = ps1.at[...].add(-(h_exp * psin))

#     #     if params.mprint > 0:
#     #         if params.iteration % params.mprint == 0:
#     #             sp_efluct1 = jnp.sqrt(rpsnorm(ps1, grids.wxyz))
#     #             sp_efluct2 = jnp.sqrt(rpsnorm(lagrange - h_exp * psin, grids.wxyz))

#     #     if forces.tbcs:
#     #         ps2 = laplace(grids, forces, 1.0, 0.0, ps1, 0, static.e0dmp)
#     #     else:
#     #         ps2 = laplace(
#     #             grids,
#     #             forces,
#     #             levels.wocc[nst],
#     #             weightuv,
#     #             ps1,
#     #             jnp.max(meanfield.v_pair[iq,...]),
#     #             static.e0dmp
#     #         )

#     #     psin = psin.at[...].add(-(static.x0dmp * ps2))
#     # else:
#     #     psin = psin.at[...].set(
#     #         (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1
#     #     )

#     # Step 4
#     ps1, hmfpsi, delpsi = hpsi01(grids, meanfield, iq, 1.0, 0.0, psin)

#     # Step 5
#     denerg = (spe_mf - spe_mf_new) / jnp.abs(spe_mf_new)

#     return psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi






# grstep_vmap = jax.vmap(jax.jit(grstep), in_axes=(None, None, None, None, None, None, 0, 0, 0, 0, 0))


# psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi = grstep_vmap(params, forces, grids, meanfield, levels, static, nst, levels.isospin, levels.sp_energy, levels.psi, levels.lagrange)
















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
    params.iteration = i
    print(params.iteration)
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
