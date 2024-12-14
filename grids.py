import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from typing import Tuple


@partial(register_dataclass,
         data_fields=['x', 'y', 'z', 'der1x', 'der2x', 'cdmpx',
                      'der1y', 'der2y', 'cdmpy', 'der1z', 'der2z', 'cdmpz'],
         meta_fields=['nx', 'ny', 'nz', 'dx', 'dy', 'dz', 'periodic', 'bangx',
                      'bangy', 'bangz', 'tbangx', 'tbangy', 'tbangz', 'tabc_x', 'tabc_y', 'tabc_z', 'wxyz'])
@dataclass
class Grids:
    """
    Class handling grid definitions and operations in 3D space.
    Implements functionality from the original Fortran module.
    """
    nx: int  # Number of points in x-direction (must be even)
    ny: int  # Number of points in y-direction (must be even) 
    nz: int  # Number of points in z-direction (must be even)
    dx: float  # Grid spacing in x-direction
    dy: float  # Grid spacing in y-direction  
    dz: float  # Grid spacing in z-direction
    bangx: float  # Bloch phase in x-direction
    bangy: float  # Bloch phase in y-direction
    bangz: float  # Bloch phase in z-direction
    tbangx: bool  # Flag for Bloch boundaries in x
    tbangy: bool  # Flag for Bloch boundaries in y
    tbangz: bool  # Flag for Bloch boundaries in z
    tabc_x: int  # Number of TABC points in x
    tabc_y: int  # Number of TABC points in y
    tabc_z: int  # Number of TABC points in z
    periodic: bool  # Flag for periodic boundaries
    wxyz: float  # Volume element (dx*dy*dz)

    x: jax.Array  # x coordinates
    y: jax.Array  # y coordinates
    z: jax.Array  # z coordinates

    der1x: jax.Array  # First derivatives in x
    der2x: jax.Array  # Second derivatives in x
    cdmpx: jax.Array  # Damping matrix in x

    der1y: jax.Array  # First derivatives in y
    der2y: jax.Array  # Second derivatives in y
    cdmpy: jax.Array  # Damping matrix in y

    der1z: jax.Array  # First derivatives in z
    der2z: jax.Array  # Second derivatives in z
    cdmpz: jax.Array  # Damping matrix in z


def sder(nmax: int, d: float) -> jnp.ndarray:
    """Calculate first derivative matrix."""
    icn = (nmax + 1) // 2
    afac = jnp.pi / icn

    i = jnp.arange(1, nmax + 1, dtype=jnp.int64)
    j = jnp.arange(1, icn, dtype=jnp.int64)[:, jnp.newaxis, jnp.newaxis]
    grid = i[jnp.newaxis, :] - i[:, jnp.newaxis]

    der = -afac * ((jnp.sum(-j * jnp.sin(j * afac * (grid)), axis=0)) - 
                   (0.5 * icn * jnp.sin(icn * afac * grid))) / (icn * d)

    return der

sder_jit = jax.jit(sder, static_argnames=['nmax', 'd'])


def sder2(nmax: int, d: float) -> jnp.ndarray:
    """Calculate second derivative matrix."""
    icn = (nmax + 1) // 2
    afac = jnp.pi / icn

    i = jnp.arange(1, nmax + 1, dtype=jnp.int64)
    j = jnp.arange(1, icn, dtype=jnp.int64)[:, jnp.newaxis, jnp.newaxis]
    grid = i[jnp.newaxis, :] - i[:, jnp.newaxis]

    der = -(afac * afac) * ((jnp.sum((j ** 2) * jnp.cos(j * afac * (grid)), axis=0)) + 
                            (0.5 * (icn ** 2) * jnp.cos(icn * afac * grid))) / (icn * d * d)

    return der

sder2_jit = jax.jit(sder2, static_argnames=['nmax', 'd'])


def init_grids(params, **kwargs) -> Grids:
    """Initialize grid with all necessary components."""
    default_kwargs = {
        'nx': 48,
        'ny': 48,
        'nz': 48,
        'dx': 0.8,
        'dy': 0.8,
        'dz': 0.8,
        'bangx': 0.0,
        'bangy': 0.0,
        'bangz': 0.0,
        'tbangx': False,
        'tbangy': False,
        'tbangz': False,
        'tabc_x': 0,
        'tabc_y': 0,
        'tabc_z': 0,
        'periodic': False
    }

    default_kwargs.update(kwargs)

    # Validate Bloch boundaries
    if not params.tfft and (abs(default_kwargs['bangx']) > 0.00001 or
                           abs(default_kwargs['bangy']) > 0.00001 or
                           abs(default_kwargs['bangz']) > 0.00001):
        raise ValueError('Bloch boundaries cannot be used without tfft.')

    # Handle TABC initialization
    if params.tabc_nprocs > 1:
        nbloch = max(1, abs(default_kwargs['tabc_x'])) * \
                 max(1, abs(default_kwargs['tabc_y'])) * \
                 max(1, abs(default_kwargs['tabc_z']))
        
        if nbloch != params.tabc_nprocs:
            raise ValueError('Number of processes not adequate for this setup of TABC')
            
        # Calculate Bloch indices
        def get_bloch_index(tabc_val: int) -> int:
            return params.tabc_myid % max(1, abs(tabc_val)) if tabc_val != 0 else 0
            
        xbloch = get_bloch_index(default_kwargs['tabc_x'])
        ybloch = (params.tabc_myid // max(1, abs(default_kwargs['tabc_x']))) % \
                 max(1, abs(default_kwargs['tabc_y'])) if default_kwargs['tabc_y'] != 0 else 0
        zbloch = (params.tabc_myid // max(1, abs(default_kwargs['tabc_x'])) // \
                 max(1, abs(default_kwargs['tabc_y']))) % \
                 max(1, abs(default_kwargs['tabc_z'])) if default_kwargs['tabc_z'] != 0 else 0
        
        # Calculate new Bloch angles
        def calc_bang(tabc: int, bloch: int) -> float:
            if tabc < 0:
                return (float(bloch) + 0.5) / float(abs(tabc))
            elif tabc > 0:
                return -1.0 + (float(bloch) + 0.5) * 2.0 / float(abs(tabc))
            return 0.0
            
        default_kwargs['bangx'] = calc_bang(default_kwargs['tabc_x'], xbloch)
        default_kwargs['bangy'] = calc_bang(default_kwargs['tabc_y'], ybloch)
        default_kwargs['bangz'] = calc_bang(default_kwargs['tabc_z'], zbloch)

    if params.tabc_nprocs == 1 and (default_kwargs['tabc_x'] != 0 or
                                   default_kwargs['tabc_y'] != 0 or
                                   default_kwargs['tabc_z'] != 0):
        raise ValueError('No TABC possible with tabc_nprocs=1.')

    # Convert Bloch angles to radians
    default_kwargs['bangx'] *= params.pi
    default_kwargs['bangy'] *= params.pi
    default_kwargs['bangz'] *= params.pi

    # Validate grid dimensions
    if any(dim % 2 != 0 for dim in [default_kwargs['nx'], default_kwargs['ny'], default_kwargs['nz']]):
        raise ValueError('Grids: nx, ny, and nz must be even')

    # Handle grid spacing
    if default_kwargs['dx'] * default_kwargs['dy'] * default_kwargs['dz'] <= 0.0:
        if default_kwargs['dx'] <= 0.0:
            raise ValueError('Grid spacing given as 0.')
        default_kwargs['dy'] = default_kwargs['dx']
        default_kwargs['dz'] = default_kwargs['dx']

    # Initialize coordinates and matrices
    for axis, n, d in [('x', default_kwargs['nx'], default_kwargs['dx']),
                      ('y', default_kwargs['ny'], default_kwargs['dy']),
                      ('z', default_kwargs['nz'], default_kwargs['dz'])]:
        default_kwargs[axis] = jnp.arange(n, dtype=jnp.float64) * d - 0.5 * (n - 1) * d
        default_kwargs[f'der1{axis}'] = sder_jit(n, d)
        default_kwargs[f'der2{axis}'] = sder2_jit(n, d)
        default_kwargs[f'cdmp{axis}'] = jnp.zeros((n, n), dtype=jnp.float64)

    default_kwargs['wxyz'] = default_kwargs['dx'] * default_kwargs['dy'] * default_kwargs['dz']

    return Grids(**default_kwargs)


def setup_damping(grid: Grids, e0dmp: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Set up damping matrices for all directions."""
    def gauss(a: jax.Array, b: jax.Array) -> jax.Array:
        """JAX implementation of Gauss elimination with pivoting."""
        n = a.shape[0]
        c = jnp.concatenate([a, b], axis=1)

        def gauss_step(k: int, c: jax.Array) -> jax.Array:
            pivot_idx = jnp.argmax(jnp.abs(c[k:, k])) + k
            c = jnp.where(k != pivot_idx,
                         c.at[k, k:].set(c[pivot_idx, k:]).at[pivot_idx, k:].set(c[k, k:]),
                         c)
            pivot = c[k, k]
            c = c.at[k, k+1:].multiply(1.0 / pivot)
            updates = c[k+1:, k:] - jnp.outer(c[k+1:, k], c[k, k+1:])
            c = c.at[k+1:, k:].set(updates)
            return c

        c = jax.lax.fori_loop(0, n, gauss_step, c)
        return c[:, n:]

    def setdmc(der2: jax.Array, h2ma: float, e0dmp: float) -> jax.Array:
        n = der2.shape[0]
        if e0dmp <= 0.0:
            return jnp.zeros((n, n))
            
        unit = -h2ma * der2 / e0dmp
        unit = unit.at[jnp.diag_indices(n)].add(1.0)
        cdmp = jnp.eye(n)
        
        return gauss(unit, cdmp)

    # Assuming h2ma is imported from Forces module
    h2ma = 1.0  # This should be imported from Forces module
    
    setdmc_jit = jax.jit(setdmc)
    cdmpx = setdmc_jit(grid.der2x, h2ma, e0dmp)
    cdmpy = setdmc_jit(grid.der2y, h2ma, e0dmp)
    cdmpz = setdmc_jit(grid.der2z, h2ma, e0dmp)
    
    return cdmpx, cdmpy, cdmpz