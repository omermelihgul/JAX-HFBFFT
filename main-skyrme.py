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

densities.rho = densities.rho.at[...].set(load4d_real('rho'))
densities.sodens = densities.sodens.at[...].set(load5d_real('sodens'))
densities.tau = densities.tau.at[...].set(load4d_real('tau'))
densities.sdens = densities.sdens.at[...].set(load5d_real('sdens'))
densities.current = densities.current.at[...].set(load5d_real('current'))
densities.chi = densities.chi.at[...].set(load4d_real('chi'))



upot = load4d_real('upot')
divaq = load4d_real('divaq')
wlspot = load5d_real('wlspot')
dbmass = load5d_real('dbmass')
spot = load5d_real('spot')
wcoul = load3d_real('wcoul')
bmass = load4d_real('bmass')
aq = load5d_real('aq')
v_pair = load4d_real('v_pair')

'''
print("vals")
print(forces.power)
print(forces.b3)
print(forces.b3p)
print(forces.b4)
print(forces.b4p)
print(forces.ex)
'''

def skyrme(
    params,
    grids,
    meanfield,
    densities,
    forces,
    static,
    coulomb
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

    # Step 2: add divergence of spin-orbit current to upot
    workden = jnp.zeros(densities.rho.shape, dtype=jnp.float64)

    for iq in range(2):
        workden = workden.at[iq,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sodens[iq,0,...]))
        workden = workden.at[iq,...].add(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sodens[iq,1,...]))
        workden = workden.at[iq,...].add(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sodens[iq,2,...]))

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].add(
            -(forces.b4 + forces.b4p) * workden[iq,...] - forces.b4 * workden[ic,...]
        )

    # Step 3: Coulomb potential
    if (params.tcoul):
        wcoul = poisson(grids, params, coulomb, densities.rho)
        meanfield.upot = meanfield.upot.at[1,...].add(wcoul)
        if (forces.ex != 0):
            meanfield.upot = meanfield.upot.at[1,...].add(
                -forces.slate * densities.rho[1,...] ** (1.0 / 3.0)
            )

    # Step 4: remaining terms of upot
    workden = workden.at[...].set(0.0)

    for iq in range(2):
        workden = workden.at[iq,...].set(jnp.einsum('ij,jkl->ikl', grids.der2x, densities.rho[iq,...]))
        workden = workden.at[iq,...].add(jnp.einsum('jl,ilk->ijk', grids.der2y, densities.rho[iq,...]))
        workden = workden.at[iq,...].add(jnp.einsum('kl,ijl->ijk', grids.der2z, densities.rho[iq,...]))

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

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.wlspot = meanfield.wlspot.at[iq,...].set(
            (forces.b4 + forces.b4p) * workvec[iq,...] + forces.b4 * workvec[ic,...]
        )

    for iq in range(2):
        workvec = workvec.at[iq,0,...].set(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sdens[iq,2,...]))
        workvec = workvec.at[iq,0,...].add(-jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq,1,...]))
        workvec = workvec.at[iq,1,...].set(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq,0,...]))
        workvec = workvec.at[iq,1,...].add(-jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq,2,...]))
        workvec = workvec.at[iq,2,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq,1,...]))
        workvec = workvec.at[iq,2,...].add(-jnp.einsum('kl,ijl->ijk', grids.der1y, densities.sdens[iq,0,...]))

    # Step 8: calculate A_q vector
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.aq = meanfield.aq.at[iq,...].set(
            -2.0 * (forces.b1 - forces.b1p) * densities.current[iq,...] - 2.0 *
            forces.b1 * densities.current[ic,...] - (forces.b4 + forces.b4p) *
            workvec[iq,...] -forces.b4 * workvec[ic,...]
        )

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

    # Step 10: combine isospin contributions
    new_spot_0 = -(forces.b4 + forces.b4p) * meanfield.spot[0,...] - forces.b4 * meanfield.spot[1,...]
    new_spot_1 = -(forces.b4 + forces.b4p) * meanfield.spot[1,...] - forces.b4 * meanfield.spot[0,...]

    meanfield.spot = meanfield.spot.at[0,...].set(new_spot_0)
    meanfield.spot = meanfield.spot.at[1,...].set(new_spot_1)

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
            raise ValueError(f"{forces.ipair=}")
        else:
            meanfield.v_pair = meanfield.v_pair.at[iq,...].set(
                v0act * densities.chi[iq,...]
            )
            meanfield.ecorrp = 0.0

    if static.outertype != 'N':
        raise ValueError(f"{static.outertype=}")

    return meanfield, wcoul




skyrme_jit = jax.jit(skyrme)
res1, res2 = skyrme_jit(params, grids, meanfield, densities, forces, static, coulomb)

print(jnp.max(jnp.abs(upot - res1.upot)))
print(jnp.max(jnp.abs(bmass - res1.bmass)))
print(jnp.max(jnp.abs(wlspot - res1.wlspot)))
print(jnp.max(jnp.abs(aq - res1.aq)))
print(jnp.max(jnp.abs(divaq - res1.divaq)))
print(jnp.max(jnp.abs(dbmass - res1.dbmass)))
print(jnp.max(jnp.abs(v_pair - res1.v_pair)))
print(jnp.max(jnp.abs(wcoul - res2)))


'''
4.547473508864641e-13
7.105427357601002e-15
7.240475580205796e-14
1.428510108366312e-14
1.742645621688738e-14
2.2026824808563106e-13
0.0
1.0658141036401503e-14
'''






#coulomb.wcoul = coulomb.wcoul.at[...].set(load3d_real('wcoul'))
#densities.rho = densities.rho.at[...].set(load4d_real('rho'))


#res = poisson_jit(grids, params, coulomb, densities.rho)





















#print(jnp.max(jnp.abs(coulomb.wcoul - res)))
