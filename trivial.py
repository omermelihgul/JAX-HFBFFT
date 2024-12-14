import jax
from jax import numpy as jnp
from typing import Tuple, Optional, Union
from functools import partial


def cmulx(xmat: jax.Array, pinn: jax.Array, pout: Optional[jax.Array] = None, 
          ifadd: int = 0) -> jax.Array:
    """Matrix multiplication of real matrix with complex wave function along x-direction."""
    # Use lax.cond instead of if statements for JIT compatibility
    def handle_output(args):
        pout, pinn = args
        return jnp.zeros_like(pinn) if pout is None else pout
    
    pout = jax.lax.cond(
        ifadd == 0,
        lambda _: jnp.zeros_like(pinn),
        handle_output,
        (pout, pinn)
    )
        
    # Perform matrix multiplication along x-axis for each spin component
    result = pout + jnp.tensordot(xmat, pinn, axes=([1], [1])).transpose(1, 0, 2, 3)
    return result


def cmuly(ymat: jax.Array, pinn: jax.Array, pout: Optional[jax.Array] = None,
          ifadd: int = 0) -> jax.Array:
    """Matrix multiplication of real matrix with complex wave function along y-direction."""
    pout = jax.lax.cond(
        ifadd == 0,
        lambda _: jnp.zeros_like(pinn),
        lambda _: jnp.zeros_like(pinn) if pout is None else pout,
        operand=None
    )
        
    # Perform matrix multiplication along y-axis for each spin component
    result = pout + jnp.tensordot(pinn, ymat, axes=([2], [0]))
    return result


def cmulz(zmat: jax.Array, pinn: jax.Array, pout: Optional[jax.Array] = None,
          ifadd: int = 0) -> jax.Array:
    """Matrix multiplication of real matrix with complex wave function along z-direction."""
    pout = jax.lax.cond(
        ifadd == 0,
        lambda _: jnp.zeros_like(pinn),
        lambda _: jnp.zeros_like(pinn) if pout is None else pout,
        operand=None
    )
        
    # Perform matrix multiplication along z-axis for each spin component
    result = pout + jnp.tensordot(pinn, zmat, axes=([3], [0]))
    return result


def rpsnorm(ps: jax.Array, wxyz: float) -> jax.Array:
    """Calculate norm of wave function."""
    return wxyz * jnp.sum(jnp.real(jnp.conjugate(ps) * ps))


def overlap(pl: jax.Array, pr: jax.Array, wxyz: float) -> jax.Array:
    """Calculate overlap between two wave functions."""
    return wxyz * jnp.sum(jnp.conjugate(pl) * pr)


def rmulx(xmat: jax.Array, finn: jax.Array, fout: Optional[jax.Array] = None,
          ifadd: int = 0) -> jax.Array:
    """Matrix multiplication of real matrix with real field along x-direction.
    
    """
    def get_output(args):
        fin, fout = args
        return jnp.zeros_like(fin) if fout is None else fout
    
    # Initialize output
    pout = jax.lax.cond(
        ifadd == 0,
        lambda x: jnp.zeros_like(x),
        lambda x: get_output((x, fout)),
        finn
    )
        
    # Perform matrix multiplication
    result = jnp.tensordot(xmat, finn, axes=([1], [0]))
    
    # Handle accumulation
    return jax.lax.cond(
        ifadd >= 0,
        lambda x: pout + x,
        lambda x: pout - x,
        result
    )

def rmuly(ymat: jax.Array, finn: jax.Array, fout: Optional[jax.Array] = None,
          ifadd: int = 0) -> jax.Array:
    """Matrix multiplication of real matrix with real field along y-direction."""
    def get_output(args):
        fin, fout = args
        return jnp.zeros_like(fin) if fout is None else fout
    
    # Initialize output
    pout = jax.lax.cond(
        ifadd == 0,
        lambda x: jnp.zeros_like(x),
        lambda x: get_output((x, fout)),
        finn
    )
        
    # Transpose to align y dimension, multiply, then transpose back
    finn_t = jnp.swapaxes(finn, 0, 1)
    result = jnp.tensordot(ymat, finn_t, axes=([1], [0]))
    result = jnp.swapaxes(result, 0, 1)
    
    # Handle accumulation
    return jax.lax.cond(
        ifadd >= 0,
        lambda x: pout + x,
        lambda x: pout - x,
        result
    )

def rmulz(zmat: jax.Array, finn: jax.Array, fout: Optional[jax.Array] = None,
          ifadd: int = 0) -> jax.Array:
    """Matrix multiplication of real matrix with real field along z-direction."""
    def get_output(args):
        fin, fout = args
        return jnp.zeros_like(fin) if fout is None else fout
    
    # Initialize output
    pout = jax.lax.cond(
        ifadd == 0,
        lambda x: jnp.zeros_like(x),
        lambda x: get_output((x, fout)),
        finn
    )
        
    # Transpose to align z dimension, multiply, then transpose back
    finn_t = jnp.moveaxis(finn, 2, 0)
    result = jnp.tensordot(zmat, finn_t, axes=([1], [0]))
    result = jnp.moveaxis(result, 0, 2)
    
    # Handle accumulation
    return jax.lax.cond(
        ifadd >= 0,
        lambda x: pout + x,
        lambda x: pout - x,
        result
    )


# JIT-compile the functions for better performance
cmulx_jit = jax.jit(cmulx)
cmuly_jit = jax.jit(cmuly)
cmulz_jit = jax.jit(cmulz)
rpsnorm_jit = jax.jit(rpsnorm)
overlap_jit = jax.jit(overlap)
rmulx_jit = jax.jit(rmulx)
rmuly_jit = jax.jit(rmuly)
rmulz_jit = jax.jit(rmulz)