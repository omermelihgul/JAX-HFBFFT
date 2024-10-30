import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass, replace
from jax.tree_util import register_dataclass

@partial(register_dataclass,
         data_fields=['hmatrix',
                      'gapmatrix',
                      'symcond',
                      'lambda_save'],
         meta_fields=['tbcs',
                      'tdiag'])
@dataclass
class Static:
    tbcs: bool
    tdiag: bool
    hmatrix: jax.Array
    gapmatrix: jax.Array
    symcond: jax.Array
    lambda_save: jax.Array


def init_static(levels, **kwargs) -> Static:
    nst = max(levels.nneut, levels.nprot)

    kwargs = {
        'tbcs': False,
        'tdiag': False,
        'hmatrix': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'gapmatrix': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'symcond': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'lambda_save': jnp.zeros((2, nst, nst), dtype=jnp.complex128)
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



def harmosc():
    pass



def grstep(
    params,
    levels,
    static,
    nst,
    iq,
    spe_mf,
    denerg,
    psin,
    lagrange
):
    sp_efluct1, sp_efluct2 = 0.0, 0.0

    # Step 1
    if static.tbcs:
        ps1, psi_mf = hpsi()
    else:
        weightuv = wguv[nst] * pairwg[nst]
        ps1, psi_mf = hpsi()

    # Step 2
    spe_mf_new = jnp.real(overlap(psin, psi_mf))

    # Step 3
    if static.e0dmp > 0.0:
        if params.iteration > 1 and not static.tbcs:
            if params.mprint > 0:
                if params.iteration % params.mprint == 0:
                    h_exp = jnp.real(overlap(psin, ps1))

            ps1 = ps1.at[...].subtract(lagrange)
        else:
            h_exp = jnp.real(overlap(psin, ps1))
            ps1 = ps1.at[...].subtract(h_exp * psin)

        if params.mprint > 0:
            if params.iteration % params.mprint == 0:
                sp_efluct1 = jnp.sqrt(rpsnorm(ps1))
                sp_efluct2 = jnp.sqrt(rpsnorm(lagrange - h_exp * psin))

        if params.tfft:
            if static.tbcs:
                ps2 = laplace()
            else:
                ps2 = laplace()
        else:
            raise NotImplementedError("Non FFT treatment not implemented for damping")

        psin = psin.at[...].subtract(
            static.x0dmp * ps2
        )
    else:
        psin = psin.at[...].multiply(
            1.0 + static.x0dmp * (spe_mf_new - spe_mf)
        )
        psin = psin.at[...].subtract(
            static.x0dmp * ps1
        )

    # Step 4
    ## call hpsi

    # Step 5
    res_denerg = (spe_mf - spe_mf_new) / jnp.abs(spe_mf_new)

    return psi_mf, spe_mf_new, res_denerg, psin














