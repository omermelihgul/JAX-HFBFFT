import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, statichf
from coulomb import init_coulomb

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

# determine wave function numbers etc
if params.nof > params.mnof:
    raise ValueError('nof > mnof.')
elif params.nof > 0:
    pass
else:
    pass

# initialize wave functions
if params.nof > 0:
    pass
elif params.nof == 0:
    print('1283')
elif params.nof == -1:
    pass
else:
    pass

# coulomb initialization
if params.tcoul:
    pass

# static or dynamic  calculation performed
statichf(params)

from test import *
from levels import cdervz02



meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
meanfield.divaq = meanfield.divaq.at[...].set(load4d_real('divaq'))
densities.sodens = densities.sodens.at[...].set(load5d_real('sodens'))
densities.current = densities.current.at[...].set(load5d_real('current'))
densities.sdens = densities.sdens.at[...].set(load5d_real('sdens'))
meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
meanfield.dbmass = meanfield.dbmass.at[...].set(load5d_real('dbmass'))
meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
densities.rho = densities.rho.at[...].set(load4d_real('rho'))
densities.tau = densities.tau.at[...].set(load4d_real('tau'))
upot_final = load4d_real('upot_final')
workden = load4d_real('workden')
coulomb.wcoul = coulomb.wcoul.at[...].set(load3d_real('wcoul'))
meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
workvec = load5d_real('workvec')
aq = load5d_real('aq')


print("vals")
print(forces.power)
print(forces.b3)
print(forces.b3p)
print(forces.b4)
print(forces.b4p)
print(forces.ex)

def skyrme(
    grids,
    meanfield,
    densities,
    forces,
    coulomb,
    params
):
    # Step 1: 3-body contribution to upot.
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].set(
            (densities.rho[0,:,:,:] + densities.rho[1,:,:,:]) ** forces.power *
            (
                (forces.b3 * (forces.power + 2.0) / 3.0 - 2.0 * forces.b3p / 3.0) *
                densities.rho[iq,:,:,:] + forces.b3 * (forces.power + 2.0) / 3.0 *
                densities.rho[ic,:,:,:] - (forces.b3p * forces.power / 3.0) *
                (densities.rho[0,:,:,:] ** 2 + densities.rho[1,:,:,:] ** 2) /
                (densities.rho[0,:,:,:] + densities.rho[1,:,:,:] + 1.0e-25)
            )
        )
    # 5.684341886080802e-14

    # Step 2: add divergence of spin-orbit current to upot
    workden = jnp.zeros(densities.rho.shape, dtype=jnp.float64)

    for iq in range(2):
        workden = workden.at[iq,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sodens[iq,0,...]))
        workden = workden.at[iq,...].add(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sodens[iq,1,...]))
        workden = workden.at[iq,...].add(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sodens[iq,2,...]))
    # 1.4246670656343705e-17

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].add(
            -(forces.b4 + forces.b4p) * workden[iq,...] - forces.b4 * workden[ic,...]
        )
    # 1.3919292422229917e-17

    # Step 3: Coulomb potential
    if (params.tcoul):
        meanfield.upot = meanfield.upot.at[1,...].add(coulomb.wcoul)
        if (forces.ex != 0):
            meanfield.upot = meanfield.upot.at[1,...].add(
                -forces.slate * densities.rho[1,...] ** (1.0 / 3.0)
            )
    # 1.1368683772161603e-13


    # Step 4: remaining terms of upot
    workden = workden.at[...].set(0.0)

    for iq in range(2):
        workden = workden.at[iq,...].set(jnp.einsum('ij,jkl->ikl', grids.der2x, densities.rho[iq,...]))
        workden = workden.at[iq,...].add(jnp.einsum('jl,ilk->ijk', grids.der2y, densities.rho[iq,...]))
        workden = workden.at[iq,...].add(jnp.einsum('kl,ijl->ijk', grids.der2z, densities.rho[iq,...]))
    # 1.3981871216373065e-15

    workvec = jnp.zeros(densities.sodens.shape, dtype=jnp.float64)

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].add(
            (forces.b0 - forces.b0p) * densities.rho[iq,...] + forces.b0 *
            densities.rho[ic,...] + (forces.b1 - forces.b1p) * densities.tau[iq,...] +
            forces.b1 * densities.tau[ic,...] - (forces.b2 - forces.b2p) *
            workden[iq,...] - forces.b2 * workden[ic,...]
        )

        # Step 5: effective mass
        meanfield.bmass = meanfield.bmass.at[iq,...].set(
            forces.h2m[iq] + (forces.b1 - forces.b1p) * densities.rho[iq,...] +
            forces.b1 * densities.rho[ic,...]
        )

        # Step 6: calculate grad(rho) and wlspot
        workvec = workvec.at[iq,0,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.rho[iq,...]))
        workvec = workvec.at[iq,1,...].set(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.rho[iq,...]))
        workvec = workvec.at[iq,2,...].set(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.rho[iq,...]))
    # 4.547473508864641e-13 upot
    # 1.5629858518551032e-15 workden
    # 0.0 bmass
    # 4.511289006746931e-16 workvec

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.wlspot = meanfield.wlspot.at[iq,...].set(
            (forces.b4 + forces.b4p) * workvec[iq,...] + forces.b4 * workvec[ic,...]
        )
    # 4.547473508864641e-13
    # 1.4502288259166107e-15
    # 0.0
    # 4.511314417735349e-16
    # 0.0 wlspot

    for iq in range(2):
        workvec = workvec.at[iq,0,...].set(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sdens[iq,2,...]))
        workvec = workvec.at[iq,0,...].add(-jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq,1,...]))
        workvec = workvec.at[iq,1,...].set(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq,0,...]))
        workvec = workvec.at[iq,1,...].add(-jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq,2,...]))
        workvec = workvec.at[iq,2,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq,1,...]))
        workvec = workvec.at[iq,2,...].add(-jnp.einsum('kl,ijl->ijk', grids.der1y, densities.sdens[iq,0,...]))
    # 4.547473508864641e-13
    # 1.4849232954361469e-15
    # 0.0
    # 1.1426388655734324e-16
    # 0.0

    # Step 8: calculate A_q vector
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.aq = meanfield.aq.at[iq,...].set(
            -2.0 * (forces.b1 - forces.b1p) * densities.current[iq,...] - 2.0 *
            forces.b1 * densities.current[ic,...] - (forces.b4 + forces.b4p) *
            workvec[iq,...] -forces.b4 * workvec[ic,...]
        )
    # 3.979039320256561e-13
    # 1.4432899320127035e-15
    # 0.0
    # 1.218051279627346e-16
    # 0.0
    # 0.0

    # Step 9: calculate the curl of the current density, store in spot
    for iq in range(2):
        meanfield.spot = meanfield.spot.at[iq,0,...].set(
            jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq,2,...])
        )
        meanfield.spot = meanfield.spot.at[iq,0,...].add(
            -jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq,1,...])
        )
        meanfield.spot = meanfield.spot.at[iq,1,...].set(
            jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq,0,...])
        )
        meanfield.spot = meanfield.spot.at[iq,1,...].add(
            -jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq,2,...])
        )
        meanfield.spot = meanfield.spot.at[iq,2,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq,1,...])
        )
        meanfield.spot = meanfield.spot.at[iq,2,...].add(
            -jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq,0,...])
        )
        # 3.979039320256561e-13 upot
        # 1.4432899320127035e-15 workden
        # 0.0 bmass
        # 1.218051279627346e-16 workvec
        # 0.0 wlspot
        # 0.0 aq
        # 0.0 spot

    # Step 10: combine isospin contributions
    rotspp = meanfield.spot[0,...]
    rotspn = meanfield.spot[1,...]
    '''
    for iq in range(2):
        meanfield.spot = meanfield.spot.at[iq,...].set(
            -(forces.b4 + forces.b4p) * rotspp -forces.
        )
    '''

    # Step 11: calculate divergence of aq in divaq
    for iq in range(2):
        meanfield.divaq = meanfield.divaq.at[iq,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, meanfield.aq[iq,0,...])
        )
        meanfield.divaq = meanfield.divaq.at[iq,...].add(
            jnp.einsum('jl,ilk->ijk', grids.der1y, meanfield.aq[iq,1,...])
        )
        meanfield.divaq = meanfield.divaq.at[iq,...].add(
            jnp.einsum('kl,ijl->ijk', grids.der1z, meanfield.aq[iq,2,...])
        )
    # 3.694822225952521e-13
    # 1.4363510381087963e-15
    # 0.0
    # 1.22532709649662e-16
    # 0.0
    # 0.0
    # 0.0
    # 0.0 divaq

    # Step 12: calculate the gradient of the effective mass in dbmass
    for iq in range(2):
        meanfield.dbmass = meanfield.dbmass.at[iq,0,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, meanfield.bmass[iq,...])
        )
        meanfield.dbmass = meanfield.dbmass.at[iq,1,...].add(
            jnp.einsum('jl,ilk->ijk', grids.der1y, meanfield.bmass[iq,...])
        )
        meanfield.dbmass = meanfield.dbmass.at[iq,2,...].add(
            jnp.einsum('kl,ijl->ijk', grids.der1z, meanfield.bmass[iq,...])
        )

    # Step 13: calculate the pairing potential, store in v_pair, and add pairing part to upot
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        if iq == 2:
            v0act, v0other, = forces.v0prot, foces.v0neut
        else:
            v0act, v0other = forces.v0neut, forces.v0prot

        if forces.ipair == 6:
            meanfield.v_pair = meanfield.v_pair.at[iq,...].set(
                v0act * densities.chi[iq,...] *
                (1.0 - (densities.rho[0,...] + densities.rho[1,...]) / foces.rho0pr)
            )
        else:
            pass


    return meanfield.upot, workden, meanfield.bmass, workvec, meanfield.wlspot, meanfield.aq, meanfield.spot, meanfield.divaq, meanfield.dbmass

# epsilon = 1.0e-25




res, res1,res2, res3, res4, res5, res6, res7, res8 = skyrme(grids, meanfield, densities, forces, coulomb, params)
print("abs")
print(jnp.max(jnp.abs(upot_final - res)))
print(jnp.max(jnp.abs(workden - res1)))
print(jnp.max(jnp.abs(meanfield.bmass - res2)))
print(jnp.max(jnp.abs(workvec - res3)))
print(jnp.max(jnp.abs(meanfield.wlspot - res4)))
print(jnp.max(jnp.abs(meanfield.aq - res5)))
print(jnp.max(jnp.abs(meanfield.spot - res6)))
print(jnp.max(jnp.abs(meanfield.divaq - res7)))
print(jnp.max(jnp.abs(meanfield.dbmass - res8)))
















'''
pinn = load4d('pinn')
meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
meanfield.aq = meanfield.aq.at[...].set(load5d_real('aq'))
meanfield.v_pair = meanfield.v_pair.at[...].set(load4d_real('v_pair'))
pout = load4d('pout')
pout_mf = load4d('pout_mf')
#pout_del = load4d('pout_del')

#pswk = load4d('pswk')
#pswk2 = load4d('pswk2')
psi = load5d('psi')
from static import grstep_vmap

size = 132
nst = jnp.arange(0, 132)
spe_mf = jnp.arange(0, 132)
iq = jnp.zeros(size, dtype=int)
iq = iq.at[size // 2:].set(1)

psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi = grstep_vmap(params, forces, grids, meanfield, levels, static, nst, iq, spe_mf, psi, psi)

print(psi_mf.shape)

import time
start_time = time.time()
for _ in range(5):
    psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi = grstep_vmap(params, forces, grids, meanfield, levels, static, nst, iq, spe_mf, psi, psi)
    psin.block_until_ready()
    psi_mf.block_until_ready()
    spe_mf_new.block_until_ready()
    denerg.block_until_ready()
    hmfpsi.block_until_ready()
    delpsi.block_until_ready()

end_time = time.time()
execution_time = (end_time - start_time) / 5

print(f"Execution time: {execution_time} seconds")
'''
'''
from meanfield import hpsi00_jit

res, res1 = hpsi00_jit(grids, meanfield, 0, 1.0, 0.0, pinn)

def relative_norm_error(res, pout):
    norm_diff = jnp.linalg.norm(res - pout)
    norm_pout = jnp.linalg.norm(pout)
    if norm_pout == 0:
        return jnp.inf
    relative_error = norm_diff / norm_pout
    return relative_error

error = relative_norm_error(res, pout)
print("Relative Norm Error:", error)

error = relative_norm_error(res1, pout_mf)
print("Relative Norm Error:", error)

#error = relative_norm_error(res2, pout_del)
#print("Relative Norm Error:", error)

print(jnp.max(jnp.abs(pout - res)))
print(jnp.max(jnp.abs(pout_mf - res1)))
#print(jnp.max(jnp.abs(pout_del - res2)))

#print(jnp.max(res1))
#print(jnp.max(pswk))
#print(jnp.max(res2))
#print(jnp.max(pswk2))
'''








'''
pinn = load4d('pinn')
pswk = load4d('pswk')
pswk2 = load4d('pswk2')
bmass = load4d_real('bmass')

res, res2 = cdervz02(grids.dz, pinn, bmass[0,...])
print(jnp.max(jnp.abs(res - pswk)))
print(jnp.max(jnp.abs(res2 - pswk2)))
'''


































# print('\n')
# print(params)
# print('\n')
# print(grids.der1x[0,:])
# print('\n')
# print(levels.isospin)



#test
#from levels import cdervz00_jit
#from test import *

#pinn = load4d('pinn')
#pswk2 = load4d('pswk2')
#pswk = load4d('pswk')
#pos_func = load3d('pos_func')

#res = cdervz00_jit(0.8, pswk2)
#omer
#print(jnp.max(jnp.abs(pswk - res)))
#print(jnp.max(pswk))
#print(jnp.max(res))
#print(jnp.max(jnp.abs(pswk2 - res2)))
#print(jnp.max(pswk2))
#print(jnp.max(res2))

#levels.psi = levels.psi.at[...].set(load5d('psi'))
#meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
#meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
#meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
#meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
#meanfield.aq = meanfield.aq.at[...].set(load5d_real('aq'))
#pout = load4d('pout')


#from meanfield import hpsi0fft, hpsi0fft_jit


#res = hpsi0fft(grids, meanfield, 0, levels.psi[0,...])

#print(jnp.max(jnp.abs(pout - res)))


#from levels import laplace


#print(laplace(grids))
#print(grids.bangx)



#ps1 = load4d('ps1')
#ps2 = load4d('ps2')

#res = laplace(grids, forces, 1.0 , 0.0, ps1, 0, 100.0)


#fft_result = jnp.fft.ifftn(psout, axes=(-3, -2, -1), norm='forward')
#print(jnp.max(jnp.abs(res - ps2)))


#difference = res[0,:,1,11] - ps2[0,:,1,11]
#norm_difference = np.linalg.norm(difference, ord=2)
#norm_exact = np.linalg.norm(ps2[0,:,1,11], ord=2)

#relative_error = norm_difference / norm_exact
#print(f"L2 Relative Norm Error: {relative_error:.4f}")

#relative_error = jnp.abs(res[0,...] - ps2[0,...]) / jnp.abs(res[0,...])
#max_relative_error = jnp.nanmax(relative_error)
#print("Maximum Relative Error:", max_relative_error)

#print(res[0,11,:,14])
#print(fft_result[0,11,:,14])
