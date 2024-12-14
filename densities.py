import jax
from jax import numpy as jnp
from jax import lax
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from typing import Optional, Tuple

@partial(register_dataclass,
         data_fields=['rho', 'chi', 'tau', 'current', 'sdens', 'sodens'],
         meta_fields=[])
         
@dataclass
class Densities:
    rho: jax.Array
    chi: jax.Array
    tau: jax.Array
    current: jax.Array
    sdens: jax.Array
    sodens: jax.Array


def init_densities(grids) -> Densities:
    """Initialize all density arrays to zero with appropriate shapes."""
    shape4d = (grids.nx, grids.ny, grids.nz, 2)
    shape5d = (grids.nx, grids.ny, grids.nz, 3, 2)

    default_kwargs = {
        'rho': jnp.zeros(shape4d, dtype=jnp.float32),
        'chi': jnp.zeros(shape4d, dtype=jnp.float32),
        'tau': jnp.zeros(shape4d, dtype=jnp.float32),
        'current': jnp.zeros(shape5d, dtype=jnp.float32),
        'sdens': jnp.zeros(shape5d, dtype=jnp.float32),
        'sodens': jnp.zeros(shape5d, dtype=jnp.float32),
    }

    return Densities(**default_kwargs)


@jax.jit
def add_density(der1x: jax.Array, der1y: jax.Array, der1z: jax.Array,
                iq: int, weight: float, weightuv: float, 
                psin: jax.Array, densities: Densities) -> Densities:
    """Add contributions from a single-particle wavefunction to all densities."""

    def real_compute(weight, weightuv, psin, densities):
        """Compute real density contributions."""
        # Non-derivative terms
        psin_conj = jnp.conjugate(psin)
        # Density term should have shape (nx, ny, nz)
        density_term = jnp.real(psin[0] * psin_conj[0] + 
                            psin[1] * psin_conj[1])
        
        # Update rho - note that iq is passed into add_density function
        rho = densities.rho.at[..., iq].add(weight * density_term)
        chi = densities.chi.at[..., iq].add(0.5 * weightuv * density_term)
        
        # Spin density components - keep grid dimensions last
        sdens = densities.sdens
        sdens = sdens.at[..., 0, iq].add(2.0 * weight * 
                                        jnp.real(psin_conj[0] * psin[1]))
        sdens = sdens.at[..., 1, iq].add(2.0 * weight * 
                                        jnp.imag(psin_conj[0] * psin[1]))
        sdens = sdens.at[..., 2, iq].add(weight * (
            jnp.real(psin_conj[0] * psin[0]) - 
            jnp.real(psin_conj[1] * psin[1])))
        
        # Process derivatives for each direction
        tau = densities.tau
        current = densities.current
        sodens = densities.sodens

        # Process x derivatives
        ps1x = jnp.stack([
            jnp.tensordot(der1x, psin[0], axes=([1], [0])),
            jnp.tensordot(der1x, psin[1], axes=([1], [0]))
        ], axis=0)
        
        tau = tau.at[..., iq].add(weight * jnp.real(
            ps1x[0] * jnp.conjugate(ps1x[0]) + 
            ps1x[1] * jnp.conjugate(ps1x[1])))
            
        current = current.at[..., 0, iq].add(weight * jnp.imag(
            ps1x[0] * jnp.conjugate(psin[0]) + 
            ps1x[1] * jnp.conjugate(psin[1])))

        # Process y derivatives
        ps1y = jnp.stack([
            jnp.tensordot(der1y, psin[0], axes=([1], [1])),
            jnp.tensordot(der1y, psin[1], axes=([1], [1]))
        ], axis=0)
        
        tau = tau.at[..., iq].add(weight * jnp.real(
            ps1y[0] * jnp.conjugate(ps1y[0]) + 
            ps1y[1] * jnp.conjugate(ps1y[1])))
            
        current = current.at[..., 1, iq].add(weight * jnp.imag(
            ps1y[0] * jnp.conjugate(psin[0]) + 
            ps1y[1] * jnp.conjugate(psin[1])))

        # Process z derivatives
        ps1z = jnp.stack([
            jnp.tensordot(der1z, psin[0], axes=([1], [2])),
            jnp.tensordot(der1z, psin[1], axes=([1], [2]))
        ], axis=0)
        
        tau = tau.at[..., iq].add(weight * jnp.real(
            ps1z[0] * jnp.conjugate(ps1z[0]) + 
            ps1z[1] * jnp.conjugate(ps1z[1])))
            
        current = current.at[..., 2, iq].add(weight * jnp.imag(
            ps1z[0] * jnp.conjugate(psin[0]) + 
            ps1z[1] * jnp.conjugate(psin[1])))

        # Spin-orbit density terms
        sodens = sodens.at[..., :, iq].add(
            weight * jnp.stack([
                jnp.real(ps1z[1] * jnp.conjugate(psin[0]) - 
                        ps1z[0] * jnp.conjugate(psin[1])),
                jnp.imag(ps1z[1] * jnp.conjugate(psin[0]) + 
                        ps1z[0] * jnp.conjugate(psin[1])),
                -jnp.real(ps1y[0] * jnp.conjugate(psin[1]) - 
                        ps1y[1] * jnp.conjugate(psin[0]))
            ], axis=-1))

        return Densities(rho=rho, chi=chi, tau=tau, 
                        current=current, sdens=sdens, sodens=sodens)

    return lax.cond(
        weight > 0,
        lambda w, wuv, p, d: real_compute(w, wuv, p, d),
        lambda w, wuv, p, d: d,
        weight, weightuv, psin, densities
    )


def update_densities(grids, iq: int, weight: float, weightuv: float, 
                    psin: jax.Array, densities: Densities) -> Densities:
    """Wrapper function to handle grid derivatives and call add_density."""
    return add_density(grids.der1x, grids.der1y, grids.der1z,
                      iq, weight, weightuv, psin, densities)