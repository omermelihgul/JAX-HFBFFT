# coulomb.py
import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from typing import Tuple, Optional
from grids import Grids  # Changed from relative to absolute import

@partial(register_dataclass,
         data_fields=['wcoul', 'q', 'nx2', 'ny2', 'nz2'],
         meta_fields=['is_initialized'])
@dataclass
class Coulomb:
    nx2: int
    ny2: int
    nz2: int
    wcoul: jax.Array
    q: jax.Array
    is_initialized: bool = False
    
    @classmethod
    def initialize(cls, grid: Grids, params) -> 'Coulomb':
        """Initialize Coulomb solver with grid parameters."""
        # Set dimensions based on boundary conditions
        nx2 = grid.nx if grid.periodic else 2 * grid.nx
        ny2 = grid.ny if grid.periodic else 2 * grid.ny
        nz2 = grid.nz if grid.periodic else 2 * grid.nz
        
        # Initialize helper arrays for momentum/position calculations
        def init_iq(n: int, d: float, periodic: bool) -> jax.Array:
            """Calculate wavenumbers or positions."""
            idx = jnp.arange(n)
            idx = jnp.where(idx <= n//2, idx, idx - n)
            
            if periodic:
                return (2.0 * params.pi * idx / (n * d)) ** 2
            else:
                return (d * idx) ** 2
        
        # Calculate coordinate contributions
        iqx = init_iq(nx2, grid.dx, grid.periodic)
        iqy = init_iq(ny2, grid.dy, grid.periodic)
        iqz = init_iq(nz2, grid.dz, grid.periodic)
        
        # Create 3D grid of q values
        i, j, k = jnp.meshgrid(iqx, iqy, iqz, indexing='ij')
        q = i + j + k
        
        # Handle origin point and create Green's function
        if grid.periodic:
            # For periodic: 1/k² with zero at origin
            q = jnp.where(q > 1e-10, 1.0 / q, 0.0)
            q = q.astype(jnp.complex64)
        else:
            # For isolated: 1/r with special value at origin
            # Use a smoother transition near origin to avoid numerical issues
            origin_val = 2.84 / (grid.dx * grid.dy * grid.dz) ** (1.0/3.0)
            q = jnp.where(q < 1e-10, 
                         origin_val,
                         1.0 / jnp.sqrt(q + 1e-10))  # Add small constant for stability
            q = jnp.fft.fftn(q)
        
        # Initialize wcoul array
        wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz))
        
        return cls(nx2=nx2, ny2=ny2, nz2=nz2,
                  wcoul=wcoul, q=q,
                  is_initialized=True)
    
    def solve_poisson(self, 
                     grid: Grids,
                     rho: jax.Array,
                     params) -> jax.Array:
        """Solve the Poisson equation using Fourier methods."""
        # Handle input density
        if grid.periodic:
            rho2 = rho
        else:
            rho2 = jnp.zeros((self.nx2, self.ny2, self.nz2), dtype=jnp.complex64)
            rho2 = rho2.at[:grid.nx, :grid.ny, :grid.nz].set(rho)
        
        # Transform to momentum space
        rho2 = jnp.fft.fftn(rho2)
        
        # Apply Green's function
        if grid.periodic:
            # Add charge factor and geometric factors for periodic case
            rho2 = 4.0 * params.pi * params.e2 * self.q * rho2
        else:
            # Multiply by e2 and volume element for isolated case
            rho2 = params.e2 * grid.wxyz * self.q * rho2
        
        # Transform back to coordinate space
        wcoul = jnp.fft.ifftn(rho2)
        
        # Extract physical region and ensure real output
        wcoul = jnp.real(wcoul[:grid.nx, :grid.ny, :grid.nz])
        
        return wcoul

def init_coulomb(grid: Grids, params) -> Coulomb:
    """Initialize Coulomb solver with given grid parameters."""
    return Coulomb.initialize(grid, params)