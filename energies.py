import jax
from jax import numpy as jnp
from jax import lax
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from typing import Optional, Tuple

@partial(register_dataclass,
         data_fields=['efluct1',
                     'efluct2',
                     'efluct1q',
                     'efluct2q',
                     'ehft',
                     'ehf0',
                     'ehf1',
                     'ehf2',
                     'ehf3',
                     'ehfls',
                     'ehflsodd',
                     'ehfc',
                     'ecorc',
                     'ehfint',
                     'tke',
                     'ehf',
                     'e3corr',
                     'e_zpe',
                     'orbital',
                     'spin',
                     'total_angmom',
                     'e_extern',
                     'ehfCrho0',
                     'ehfCrho1',
                     'ehfCdrho0',
                     'ehfCdrho1',
                     'ehfCtau0',
                     'ehfCtau1',
                     'ehfCdJ0',
                     'ehfCdJ1',
                     'ehfCj0',
                     'ehfCj1'],
         meta_fields=[])
@dataclass
class Energies:
    # Fluctuation measures
    efluct1: jax.Array  # Maximum absolute value of λ⁻_αβ
    efluct2: jax.Array  # RMS of λ⁻_αβ
    efluct1q: jax.Array  # Maximum absolute value of λ⁻_αβ for isospin q
    efluct2q: jax.Array  # RMS of λ⁻_αβ for isospin q
    
    # Energy components
    ehft: jax.Array      # Kinetic energy
    ehf0: jax.Array      # t0 contribution
    ehf1: jax.Array      # b1 contribution (current part)
    ehf2: jax.Array      # b2 contribution (Laplacian part)
    ehf3: jax.Array      # t3 contribution (density dependent)
    ehfls: jax.Array     # Spin-orbit contribution (time even)
    ehflsodd: jax.Array  # Spin-orbit contribution (odd-odd)
    ehfc: jax.Array      # Coulomb contribution
    ecorc: jax.Array     # Slater & Koopman exchange
    ehfint: jax.Array    # Integrated total energy
    tke: jax.Array       # Kinetic energy summed
    ehf: jax.Array       # Total energy from s.p. levels
    e3corr: jax.Array    # Rearrangement energy 3-body term
    e_zpe: jax.Array     # c.m. energy correction
    
    # Angular momentum components
    orbital: jax.Array     # Total orbital angular momentum components
    spin: jax.Array        # Total spin components 
    total_angmom: jax.Array  # Total angular momentum components
    
    # External field energy
    e_extern: jax.Array
    
    # Additional energy terms
    ehfCrho0: jax.Array   # C^ρ_0 contribution
    ehfCrho1: jax.Array   # C^ρ_1 contribution
    ehfCdrho0: jax.Array  # C^∇ρ_0 contribution
    ehfCdrho1: jax.Array  # C^∇ρ_1 contribution
    ehfCtau0: jax.Array   # C^τ_0 contribution
    ehfCtau1: jax.Array   # C^τ_1 contribution
    ehfCdJ0: jax.Array    # C^∇J_0 contribution
    ehfCdJ1: jax.Array    # C^∇J_1 contribution
    ehfCj0: jax.Array     # C^j_0 contribution
    ehfCj1: jax.Array     # C^j_1 contribution


def init_energies(force_params=None, **kwargs) -> Energies:
    """Initialize energy arrays with default or specified values."""
    default_kwargs = {
        'efluct1': jnp.zeros(1, dtype=jnp.float64),
        'efluct2': jnp.zeros(1, dtype=jnp.float64),
        'efluct1q': jnp.zeros(2, dtype=jnp.float64),
        'efluct2q': jnp.zeros(2, dtype=jnp.float64),
        'ehft': jnp.zeros(1, dtype=jnp.float64),
        'ehf0': jnp.zeros(1, dtype=jnp.float64),
        'ehf1': jnp.zeros(1, dtype=jnp.float64),
        'ehf2': jnp.zeros(1, dtype=jnp.float64),
        'ehf3': jnp.zeros(1, dtype=jnp.float64),
        'ehfls': jnp.zeros(1, dtype=jnp.float64),
        'ehflsodd': jnp.zeros(1, dtype=jnp.float64),
        'ehfc': jnp.zeros(1, dtype=jnp.float64),
        'ecorc': jnp.zeros(1, dtype=jnp.float64),
        'ehfint': jnp.zeros(1, dtype=jnp.float64),
        'tke': jnp.zeros(1, dtype=jnp.float64),
        'ehf': jnp.zeros(1, dtype=jnp.float64),
        'e3corr': jnp.zeros(1, dtype=jnp.float64),
        'e_zpe': jnp.zeros(1, dtype=jnp.float64),
        'orbital': jnp.zeros(3, dtype=jnp.float64),
        'spin': jnp.zeros(3, dtype=jnp.float64),
        'total_angmom': jnp.zeros(3, dtype=jnp.float64),
        'e_extern': jnp.zeros(1, dtype=jnp.float64),
        'ehfCrho0': jnp.zeros(1, dtype=jnp.float64),
        'ehfCrho1': jnp.zeros(1, dtype=jnp.float64),
        'ehfCdrho0': jnp.zeros(1, dtype=jnp.float64),
        'ehfCdrho1': jnp.zeros(1, dtype=jnp.float64),
        'ehfCtau0': jnp.zeros(1, dtype=jnp.float64),
        'ehfCtau1': jnp.zeros(1, dtype=jnp.float64),
        'ehfCdJ0': jnp.zeros(1, dtype=jnp.float64),
        'ehfCdJ1': jnp.zeros(1, dtype=jnp.float64),
        'ehfCj0': jnp.zeros(1, dtype=jnp.float64),
        'ehfCj1': jnp.zeros(1, dtype=jnp.float64),
    }

    default_kwargs.update(kwargs)
    return Energies(**default_kwargs)


@jax.jit
def compute_integrated_energy(grids, densities, force_params, coulomb_field=None, mass_number=None):
    """Compute the integrated energy from densities."""
    
    # Helper function to compute derivatives
    def compute_laplacian(field, grids):
        return (jnp.tensordot(grids.der2x, field, axes=([1], [0])) +
                jnp.tensordot(grids.der2y, field, axes=([1], [1])) +
                jnp.tensordot(grids.der2z, field, axes=([1], [2])))
    
    # Step 1: Compute laplacians and basic density terms
    rho_total = densities.rho[..., 0] + densities.rho[..., 1]
    d2rho = jnp.stack([compute_laplacian(densities.rho[..., i], grids) 
                       for i in range(2)], axis=-1)
    d2rho_total = d2rho[..., 0] + d2rho[..., 1]
    
    # Compute energy terms
    ehf0 = grids.wxyz * jnp.sum(
        force_params.b0 * rho_total**2 - 
        force_params.b0p * (densities.rho[..., 0]**2 + densities.rho[..., 1]**2)
    ) / 2.0
    
    ehf2 = grids.wxyz * jnp.sum(
        -force_params.b2 * rho_total * d2rho_total +
        force_params.b2p * (densities.rho[..., 0] * d2rho[..., 0] + 
                           densities.rho[..., 1] * d2rho[..., 1])
    ) / 2.0
    
    ehf3 = grids.wxyz * jnp.sum(
        rho_total**force_params.power * (
            force_params.b3 * rho_total**2 -
            force_params.b3p * (densities.rho[..., 0]**2 + densities.rho[..., 1]**2)
        )
    ) / 3.0
    
    # Compute kinetic energy
    ehft = grids.wxyz * jnp.sum(
        force_params.h2m[0] * densities.tau[..., 0] +
        force_params.h2m[1] * densities.tau[..., 1]
    )
    
    # Compute spin-orbit terms
    div_j = jnp.stack([
        jnp.sum([jnp.tensordot(grids.der1x, densities.sodens[..., 0, i], axes=([1], [0])),
                 jnp.tensordot(grids.der1y, densities.sodens[..., 1, i], axes=([1], [1])),
                 jnp.tensordot(grids.der1z, densities.sodens[..., 2, i], axes=([1], [2]))],
                axis=0) for i in range(2)
    ], axis=-1)
    
    ehfls = -grids.wxyz * jnp.sum(
        force_params.b4 * rho_total * jnp.sum(div_j, axis=-1) +
        force_params.b4p * (densities.rho[..., 0] * div_j[..., 0] +
                           densities.rho[..., 1] * div_j[..., 1])
    )
    
    # Compute Coulomb energy if field is provided
    ehfc = jnp.where(
        coulomb_field is not None,
        grids.wxyz * jnp.sum(0.5 * densities.rho[..., 1] * coulomb_field),
        0.0
    )
    
    # Optional center of mass correction
    e_zpe = jnp.where(
        mass_number is not None,
        17.3 / mass_number**0.2,
        0.0
    )
    
    return ehf0, ehf2, ehf3, ehft, ehfls, ehfc, e_zpe


@jax.jit
def sum_energy(sp_states, force_params, epair):
    """Compute energy from single-particle states."""
    wocc = sp_states.occupations
    wstates = sp_states.wavefunctions
    sp_energy = sp_states.energies
    sp_kinetic = sp_states.kinetic_energy
    
    # Compute total energy
    ehf = (jnp.sum(wocc * wstates * (sp_kinetic + sp_energy)) / 2.0 +
           force_params.e3corr + force_params.ecorrp + force_params.ecorc -
           epair[0] - epair[1] - force_params.e_zpe)
    
    # Compute kinetic energy
    tke = jnp.sum(wocc * wstates * sp_kinetic)
    
    # Compute angular momenta
    orbital = jnp.sum(wocc * wstates * sp_states.orbital, axis=1)
    spin = jnp.sum(wocc * wstates * sp_states.spin, axis=1)
    total_angmom = orbital + spin
    
    return ehf, tke, orbital, spin, total_angmom