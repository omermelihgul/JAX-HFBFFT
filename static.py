import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass, replace
from jax.tree_util import register_dataclass
from meanfield import hpsi00, hpsi01
from trivial import rpsnorm, overlap
from levels import laplace

@partial(register_dataclass,
         data_fields=['hmatrix',
                      'gapmatrix',
                      'symcond',
                      'lambda_save'],
         meta_fields=['e0dmp', 'x0dmp', 'tdiag', 'outertype'])
@dataclass
class Static:
    tdiag: bool
    hmatrix: jax.Array
    gapmatrix: jax.Array
    symcond: jax.Array
    lambda_save: jax.Array
    e0dmp: float
    x0dmp: float
    outertype: str


def init_static(levels, **kwargs) -> Static:
    nst = max(levels.nneut, levels.nprot)

    kwargs = {
        'tdiag': False,
        'hmatrix': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'gapmatrix': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'symcond': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'lambda_save': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'e0dmp': 100.0,
        'x0dmp': 0.2,
        'outertype': 'N'
    }

    return Static(**kwargs)


def diagstep(grids, forces, static, levels, energies, diagonalize=False, construct=True):
    for iq in range(2):
        start, end = (0, levels.nneut) if iq == 0 else (levels.nneut, levels.nneut + levels.nprot)
        nst = end - start
        shape = (nst, -1)
        shape5d = (nst, grids.nx, grids.ny, grids.nz, 2)

        psi_2d = levels.psi.at[start:end,...].get().reshape(shape, order='F')
        hampsi_2d = levels.hampsi.at[start:end,...].get().reshape(shape, order='F')

        rhomatr_lin = jnp.dot(jnp.conjugate(psi_2d), psi_2d.T) * grids.wxyz

        if diagonalize:
            lambda_lin = jnp.dot(jnp.conjugate(psi_2d), hampsi_2d.T) * grids.wxyz

        if forces.tbcs:
            x = jnp.sqrt(jnp.sum(levels.wocc.at[start:end].get() *
                                 levels.wstates.at[start:end].get() *
                                 levels.sp_efluct1.at[start:end].get() ** 2) /
                         jnp.sum(levels.wocc.at[start:end].get() *
                                 levels.wstates.at[start:end].get()))
            energies.efluct1q = energies.efluct1q.at[iq].set(x)

        levels.sp_norm = levels.sp_norm.at[start:end].set(jnp.real(jnp.diagonal(rhomatr_lin)))

        if diagonalize:
            _, unitary_lam = jnp.linalg.eigh(lambda_lin, symmetrize_input=False)

        w, v = jnp.linalg.eigh(rhomatr_lin, symmetrize_input=False)
        unitary_rho = jnp.dot(v, jnp.dot(jnp.diag(jnp.sqrt(1.0 / w)), jnp.conjugate(v.T)))

        if diagonalize:
            unitary = jnp.dot(unitary_rho, unitary_lam)
        else:
            unitary = unitary_rho

        # Step 6: recombine
        psi_temp_2d = jnp.dot(psi_2d.T, unitary).T
        levels.psi = levels.psi.at[start:end,...].set(psi_temp_2d.reshape(shape5d, order='F'))

        # Step 7: recalculate
        psi_2d = levels.psi.at[start:end,...].get().reshape(shape, order='F')

        if construct:
            hmfpsi_2d = levels.hmfpsi.at[start:end,...].get().reshape(shape, order='F')

            lambda_lin = jnp.dot(jnp.conjugate(psi_2d), hmfpsi_2d.T) * grids.wxyz
            static.hmatrix = static.hmatrix.at[iq,0:nst,0:nst].set(jnp.dot(lambda_lin, unitary))

            if forces.ipair != 0:
                delpsi_2d = levels.delpsi.at[start:end,...].get().reshape(shape, order='F')

                lambda_lin = jnp.dot(jnp.conjugate(psi_2d), delpsi_2d.T) * grids.wxyz
                static.gapmatrix = static.gapmatrix.at[iq,0:nst,0:nst].set(jnp.dot(lambda_lin, unitary))

            # update sp_energy and deltaf
            temp_diag = jnp.real(jnp.diagonal(static.hmatrix.at[iq,0:nst,0:nst].get()))
            levels.sp_energy = levels.sp_energy.at[start:end].set(temp_diag)

            if forces.ipair != 0:
                temp_diag = jnp.real(jnp.diagonal(static.gapmatrix.at[iq,0:nst,0:nst].get()))
                levels.deltaf = levels.deltaf.at[start:end].set(temp_diag)
                levels.deltaf = levels.deltaf.at[start:end].multiply(levels.pairwg.at[start:end].get())

            if static.tbcs:
                matrix = jnp.diag(static.gapmatrix.at[iq,0:nst,0:nst].get())
                static.gapmatrix = static.gapmatrix.at[iq,0:nst,0:nst].set(matix)

            # calculate lambda and lagrange
            weight = jnp.multiply(levels.wocc.at[start:end].get(), levels.wstates.at[start:end].get())
            weightuv = (levels.wguv.at[start:end].get() *
                        levels.pairwg.at[start:end].get() *
                        levels.wstates.at[start:end].get())

            lambda_temp = (weight[:, jnp.newaxis] *
                           static.hmatrix.at[iq,0:nst,0:nst].get() -
                           weightuv[:, jnp.newaxis] *
                           static.gapmatrix.at[iq,0:nst,0:nst].get())

            cmplxhalf = 0.5 + 0.5j
            lambda_lin = cmplxhalf * (lambda_temp - jnp.transpose(jnp.conj(lambda_temp)))

            energies.efluct1q = energies.efluct1q.at[iq].set(jnp.max(jnp.abs(lambda_lin)))
            convergence2 = jnp.sqrt(jnp.sum(jnp.real(lambda_lin)**2 + jnp.imag(lambda_lin)**2) / nst**2)
            energies.efluct2q = energies.efluct1q.at[iq].set(convergence2)

            static.symcond = static.symcond.at[iq,0:nst,0:nst].set(lambda_lin)
            lambda_lin = cmplxhalf * (lambda_temp + jnp.transpose(jnp.conj(lambda_temp)))

            # recombine
            lagrange_2d = jnp.dot(psi_2d.T, lambda_lin).T
            levels.lagrange = levels.lagrange.at[start:end,...].set(lagrange_2d.reshape(shape5d, order='F'))
            static.lambda_save = static.lambda_save.at[iq,0:nst,0:nst].set(lambda_lin)

            energies.efluct1 = energies.efluct1.at[0].set(jnp.max(energies.efluct1q))
            energies.efluct2 = energies.efluct2.at[0].set(jnp.average(energies.efluct2q))
        else:
            lagrange_2d = jnp.dot(psi_2d.T, static.lambda_save.at[iq,0:nst,0:nst].get()).T
            levels.lagrange = levels.lagrange.at[start:end,...].set(lagrange_2d.reshape(shape5d, order='F'))

    return static, levels, energies

diagstep_jit = jax.jit(diagstep, static_argnames=['diagonalize', 'construct'])

'''
def statichf(params):
    firstiter = 0

    # step 1: initialization
    if params.trestart:
        firstiter = params.iteration + 1
    else:
        firstiter = 1
        print('initial orthogonalization...')

        print('done')

    # step 2: calculate densities and mean field
    print('initial add density...')
    print('done')

    # step 3: initial gradient step

    # step 4: start static iteration loop

def statichf(params):
    taddnew = True
    addnew = 0.2
    addco = 1.0 - addnew

    firstiter = 0
    # Step 4: start static iteration loop
    tbcssav = forces.tbcs # save whether we do HF+BCS

    for i in range(firstiter, static.maxiter + 1):
        # Do HF+BCS in first inibcs iterations
        if i <= static.inibcs:
            forces.tbcs = True
        else:
            forces.tbcs = tbcssav

        # Diagonalize matrix lambda (or matrix h_mf
        # for HF+BCS) in first inidiag iterations
        if i > static.inidiag:
            static.tdiag = False
        else:
            static.tdiag = True

        # Annealing: enhance pairing strengths
        # in first iteranneal iterations
        if static.iteranneal > 0:
            if i < static.iteranneal:
                pass

        # Compute expectation value of constraint
        if constraint.tconstraint:
            raise ValueError(f"{constraint.tconstraint=}")

        print(f"Static Iteration No. {i:6d}") if params.wflag else None

        # Step 5: gradient step
        static.delesum = 0.0
        static.sumflu = 0.0
        levels.sp_efluct1 = levels.sp_efluct1.at[...].set(0.0)
        levels.sp_efluct2 = levels.sp_efluct2.at[...].set(0.0)

        # grstep

        # Step 6: diagonalization and orthonormalization
        # sp_energy and deltaf will also be updated in diagstep
        if forces.tbcs:
            static.tdiag = True

        print(f"Iteration number, tdiag = {iter:6d} {static.tdiag}") if params.wflag else None

        # diagstep

        if (forces.ipair != 0 and
            static.iternat > 0 and
            i > static.iternat_start and
            not forces.tbcs
        ):
            raise ValueError(f"{forces.ipair=}")

        # Step 8: get new densities and fields with relaxation
        if taddnew:
            meanfield.upot = meanfield.upot.at[...].set(densities.rho)
            meanfield.bmass = meanfield.bmass.at[...].set(densities.tau)
            meanfield.v_pair = meanfield.v_pair.at[...].set(densities.chi)

        densities.rho = densities.rho.at[...].set(0.0)
        densities.tau = densities.tau.at[...].set(0.0)
        densities.chi = densities.chi.at[...].set(0.0)
        densities.current = densities.current.at[...].set(0.0)
        densities.sdens = densities.sdens.at[...].set(0.0)
        densities.sodens = densities.sodens.at[...].set(0.0)

        # add_density

        # Step 8a: optional constraint step
        if constraint.tconstraint:
            raise ValueError(f"{constraint.tconstraint=}")

        # Linear mixing of old and new densities
        if taddnew:
            densities.rho = densities.rho.at[...].set(
                addnew * densities.rho + addco * meanfield.upot
            )
            densities.tau = densities.tau.at[...].set(
                addnew * densities.tau + addco * meanfield.bmass
            )
            densities.chi = densities.chi.at[...].set(
                addnew * densities.chi + addco * meanfield.v_pair
            )

        # Step 8b: construct potentials

        # skyrme
        if constraint.tconstraint:
            raise ValueError(f"{constraint.tconstraint=}")

        # Calculate and print information
        if static.tsort:
            raise ValueError(f"{static.tsort=}")

        # sp_properties

        if params.mprint > 0:
            # sinfo
            pass

        # Step 9: check for convergence, saving wave functions
        if (energies.efluct1 < static.serr and
            i > 1 and
            not ttabc

        ):
            # CALL write_wavefunctions
            break
        if params.mrest > 0:
            # CALL write_wavefunctions
            raise ValueError(f"{params.mrest=}")

        # Step 10: calculate new step size
        if static.tvaryx_0:
            pass



    pass


'''



def harmosc():
    pass



def grstep(
    params,
    forces,
    grids,
    meanfield,
    levels,
    static,
    nst,
    iq,
    spe_mf,
    psin,
    lagrange
):
    sp_efluct1, sp_efluct2 = 0.0, 0.0

    # Step 1
    if forces.tbcs:
        ps1, psi_mf = hpsi00(grids, meanfield, iq, 1.0, 0.0, psin)
    else:
        weightuv = levels.wguv[nst] * levels.pairwg[nst]
        ps1, psi_mf = hpsi00(
            grids,
            meanfield,
            iq,
            levels.wstates[nst] * levels.wocc[nst],
            levels.wstates[nst] * weightuv,
            psin
        )

    # Step 2
    spe_mf_new = jnp.real(overlap(psin, psi_mf, grids.wxyz))

    # Step 3
    if static.e0dmp > 0.0:
        if params.iteration > 1 and not forces.tbcs:
            if params.mprint > 0:
                if params.iteration % params.mprint == 0:
                    h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))

            ps1 = ps1.at[...].add(-lagrange)
        else:
            h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))
            ps1 = ps1.at[...].add(-(h_exp * psin))

        if params.mprint > 0:
            if params.iteration % params.mprint == 0:
                sp_efluct1 = jnp.sqrt(rpsnorm(ps1, grids.wxyz))
                sp_efluct2 = jnp.sqrt(rpsnorm(lagrange - h_exp * psin, grids.wxyz))

        if params.tfft:
            if forces.tbcs:
                ps2 = laplace(grids, forces, 1.0, 0.0, ps1, 0, static.e0dmp)
            else:
                ps2 = laplace(
                    grids,
                    forces,
                    levels.wocc[nst],
                    weightuv,
                    ps1,
                    jnp.max(meanfield.v_pair[iq,...]),
                    static.e0dmp
                )
        else:
            raise NotImplementedError("Non FFT treatment not implemented for damping")

        psin = psin.at[...].add(-(static.x0dmp * ps2))
    else:
        psin = psin.at[...].set(
            (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1
        )

    # Step 4
    ps1, hmfpsi, delpsi = hpsi01(grids, meanfield, iq, 1.0, 0.0, psin)

    # Step 5
    denerg = (spe_mf - spe_mf_new) / jnp.abs(spe_mf_new)

    return psin, psi_mf, spe_mf_new, denerg, hmfpsi, delpsi


grstep_vmap = jax.vmap(jax.jit(grstep), in_axes=(None, None, None, None, None, None, 0, 0, 0, 0, 0))












