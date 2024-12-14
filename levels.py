import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from typing import Tuple, Optional

@partial(register_dataclass,
         data_fields=['psi', 'hampsi', 'lagrange', 'hmfpsi', 'delpsi', 'psi_temp', 'sp_orbital',
                     'sp_spin', 'isospin', 'sp_energy', 'sp_efluct1', 'sp_kinetic',
                     'sp_norm', 'sp_efluct2', 'sp_parity', 'wocc', 'wguv', 'pairwg',
                     'wstates', 'deltaf', 'charge_number', 'mass_number'],
         meta_fields=['nstmax', 'nneut', 'nprot', 'npsi', 'npmin', 'nstloc'])
@dataclass
class Levels:
    """
    Class handling wave function data and operations.
    Implements functionality from the original Fortran module.
    """
    nstmax: int  # Total number of wave functions
    nstloc: int  # Number of wave functions on current node
    nneut: int   # Physical number of neutrons
    nprot: int   # Physical number of protons
    npsi: jax.Array  # Wave function indices per particle type
    npmin: jax.Array  # Starting indices per particle type

    charge_number: float  # Physical charge number
    mass_number: float   # Physical mass number

    # Wave function arrays
    psi: jax.Array      # Main wave function array
    hampsi: jax.Array   # Hamiltonian applied to wave functions
    lagrange: jax.Array # Lagrange corrections
    hmfpsi: jax.Array   # Mean-field Hamiltonian applied to wave functions
    delpsi: jax.Array   # Pairing potential applied to wave functions
    psi_temp: jax.Array # Temporary storage for wavefunctions

    # Single particle properties
    sp_orbital: jax.Array  # Orbital angular momentum components
    sp_spin: jax.Array     # Spin components
    isospin: jax.Array     # Isospin values (1=neutron, 2=proton)
    sp_energy: jax.Array   # Single-particle energies
    sp_efluct1: jax.Array  # Energy fluctuations (method 1)
    sp_kinetic: jax.Array  # Kinetic energies
    sp_norm: jax.Array     # Wave function norms
    sp_efluct2: jax.Array  # Energy fluctuations (method 2)
    sp_parity: jax.Array   # Parity values
    
    # Occupation and pairing
    wocc: jax.Array    # Pairing occupation probabilities
    wguv: jax.Array    # Pairing uv values
    pairwg: jax.Array  # Soft cutoff on pairing gap
    wstates: jax.Array # Soft cutoff on sum over states
    deltaf: jax.Array  # Effective single-particle pairing gaps

def init_levels(grids, **kwargs) -> Levels:
    """Initialize level arrays with default values."""
    default_kwargs = {
        'nneut': kwargs.get('nneut', 82),
        'nprot': kwargs.get('nprot', 50),
        'nstloc': kwargs.get('nstloc', None),  # Will be set to nstmax if not provided
    }

    default_kwargs['nstmax'] = default_kwargs['nneut'] + default_kwargs['nprot']
    if default_kwargs['nstloc'] is None:
        default_kwargs['nstloc'] = default_kwargs['nstmax']

    # Initialize particle indices
    default_kwargs['npsi'] = jnp.array([default_kwargs['nneut'], default_kwargs['nprot']])
    default_kwargs['npmin'] = jnp.array([1, default_kwargs['nneut'] + 1])

    default_kwargs['charge_number'] = float(default_kwargs['nprot'])
    default_kwargs['mass_number'] = float(default_kwargs['nstmax'])

    # Initialize wave function arrays
    shape5d = (default_kwargs['nstloc'], 2, grids.nx, grids.ny, grids.nz)
    shape2d = (default_kwargs['nstmax'], 3)
    
    default_kwargs['psi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['hampsi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['lagrange'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['hmfpsi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['delpsi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['psi_temp'] = jnp.zeros(shape5d, dtype=jnp.complex128)

    # Initialize single particle properties
    default_kwargs['sp_orbital'] = jnp.zeros(shape2d, dtype=jnp.float64)
    default_kwargs['sp_spin'] = jnp.zeros(shape2d, dtype=jnp.float64)

    # Initialize isospin array - 1 for neutrons, 2 for protons
    isospin = jnp.ones(default_kwargs['nstmax'], dtype=jnp.int32)
    isospin = isospin.at[default_kwargs['nneut']:].set(2)
    default_kwargs['isospin'] = isospin

    # Initialize 1D arrays
    for key in ['sp_energy', 'sp_efluct1', 'sp_kinetic', 'sp_norm', 
                'sp_efluct2', 'sp_parity', 'wocc', 'wguv', 'deltaf']:
        default_kwargs[key] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)

    # Initialize arrays with ones
    default_kwargs['pairwg'] = jnp.ones(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['wstates'] = jnp.ones(default_kwargs['nstmax'], dtype=jnp.float64)

    return Levels(**default_kwargs)


def cdervx02(dx: float, psin: jnp.ndarray, pos_func: jnp.ndarray):
    iq, nx, ny, nz = psin.shape
    kfac = (jnp.pi + jnp.pi) / (dx * nx)

    d1psout = jnp.fft.fft(psin, axis=1, norm="backward")

    idx = jnp.arange(0, nx//2)
    idx_4d = idx[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,idx,:,:].multiply(-(idx_4d*kfac)**2/nx)
    d2psout = d2psout.at[:,nx-1-idx,:,:].multiply(-((idx_4d+1)*kfac)**2/nx)

    idx = jnp.arange(0, nx//2)
    idx_4d = idx[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d1psout_temp = jnp.copy(d1psout[:,nx//2,:,:])
    d1psout = d1psout.at[:,0,:,:].set(0.0)
    d1psout = d1psout.at[:,idx,:,:].multiply((1j * idx_4d) * kfac / nx)
    d1psout = d1psout.at[:,ny-idx,:,:].multiply(-((1j * idx_4d) * kfac) / nx)

    d1psout = d1psout.at[:,nx//2,:,:].set(0.0)

    d1psout = jnp.fft.ifft(d1psout, axis=1, norm="forward")

    d2psout_temp = jnp.multiply(d1psout, pos_func)
    d2psout = jnp.fft.fft(d2psout_temp, axis=1, norm="backward")
    d2psout = d2psout.at[:,idx,:,:].multiply(((1j * idx_4d)*kfac/nx))
    d2psout = d2psout.at[:,ny-idx,:,:].multiply((-((1j * idx_4d)*kfac)/nx))
    d2psout = d2psout.at[:,0,:,:].set(0.0)
    pos_func_ave = jnp.mean(pos_func, axis=0)
    d2psout = d2psout.at[:,nx//2,:,:].set(-(jnp.pi / dx) ** 2 * pos_func_ave * d1psout_temp / nx)

    d2psout = jnp.fft.ifft(d2psout, axis=1, norm="forward")

    return d1psout, d2psout

def cdervy02(dy: float, psin: jnp.ndarray, pos_func: jnp.ndarray):
    iq, nx, ny, nz = psin.shape
    kfac = (jnp.pi + jnp.pi) / (dy * ny)

    d1psout = jnp.fft.fft(psin, axis=2, norm="backward")

    idy = jnp.arange(0, ny//2)
    idy_4d = idy[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,idy,:].multiply(-(idy_4d*kfac)**2/ny)
    d2psout = d2psout.at[:,:,ny-1-idy,:].multiply(-((idy_4d+1)*kfac)**2/ny)

    idy = jnp.arange(1, ny//2)
    idy_4d = idy[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]

    d1psout_temp = jnp.copy(d1psout[:,:,ny//2,:])
    d1psout = d1psout.at[:,:,0,:].set(0.0)
    d1psout = d1psout.at[:,:,idy,:].multiply((1j * idy_4d) * kfac / ny)
    d1psout = d1psout.at[:,:,ny-idy,:].multiply(-((1j * idy_4d) * kfac) / ny)

    d1psout = d1psout.at[:,:,ny//2,:].set(0.0)

    d1psout = jnp.fft.ifft(d1psout, axis=2, norm="forward")

    d2psout_temp = jnp.multiply(d1psout, pos_func)
    d2psout = jnp.fft.fft(d2psout_temp, axis=2, norm="backward")
    d2psout = d2psout.at[:,:,idy,:].multiply(((1j * idy_4d)*kfac/ny))
    d2psout = d2psout.at[:,:,ny-idy,:].multiply((-((1j * idy_4d)*kfac)/ny))
    d2psout = d2psout.at[:,:,0,:].set(0.0)
    pos_func_ave = jnp.mean(pos_func, axis=1)
    d2psout = d2psout.at[:,:,ny//2,:].set(-(jnp.pi / dy) ** 2 * pos_func_ave * d1psout_temp / ny)

    d2psout = jnp.fft.ifft(d2psout, axis=2, norm="forward")

    return d1psout, d2psout

def cdervz02(dz: float, psin: jnp.ndarray, pos_func: jnp.ndarray):
    iq, nx, ny, nz = psin.shape
    kfac = (jnp.pi + jnp.pi) / (dz * nz)

    d1psout = jnp.fft.fft(psin, axis=3, norm="backward")

    idz = jnp.arange(0, nz//2)
    idz_4d = idz[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,:,idz].multiply(-(idz_4d*kfac)**2/nz)
    d2psout = d2psout.at[:,:,:,nz-1-idz].multiply(-((idz_4d+1)*kfac)**2/nz)

    idz = jnp.arange(1, nz//2)
    idz_4d = idz[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d1psout_temp = jnp.copy(d1psout[:,:,:,nz//2])
    d1psout = d1psout.at[:,:,:,0].set(0.0)
    d1psout = d1psout.at[:,:,:,idz].multiply((1j * idz_4d) * kfac / nz)
    d1psout = d1psout.at[:,:,:,nz-idz].multiply(-((1j * idz_4d) * kfac) / nz)

    d1psout = d1psout.at[:,:,:,nz//2].set(0.0)

    d1psout = jnp.fft.ifft(d1psout, axis=3, norm="forward")

    d2psout_temp = jnp.multiply(d1psout, pos_func)
    d2psout = jnp.fft.fft(d2psout_temp, axis=3, norm="backward")
    d2psout = d2psout.at[:,:,:,idz].multiply(((1j * idz_4d)*kfac/nz))
    d2psout = d2psout.at[:,:,:,nz-idz].multiply((-((1j * idz_4d)*kfac)/nz))
    d2psout = d2psout.at[:,:,:,0].set(0.0)
    pos_func_ave = jnp.mean(pos_func, axis=2)
    d2psout = d2psout.at[:,:,:,nz//2].set(-(jnp.pi / dz) ** 2 * pos_func_ave * d1psout_temp / nz)

    d2psout = jnp.fft.ifft(d2psout, axis=3, norm="forward")

    return d1psout, d2psout


@jax.jit
def laplace(grids, forces, wg, wguv, psin, v_pairmax, e0inv):
    if grids.bangx > 0.000001 or grids.bangy > 0.000001 or grids.bangz > 0.000001:
        raise ValueError('Laplace does not work with Bloch boundaries.')

    weightmin = jnp.array([0.1])
    weight = jnp.maximum(wg, weightmin)
    weightuv = jnp.maximum(wguv, weightmin)

    kfacx = (jnp.pi + jnp.pi) / (grids.dx * grids.nx)
    kfacy = (jnp.pi + jnp.pi) / (grids.dy * grids.ny)
    kfacz = (jnp.pi + jnp.pi) / (grids.dz * grids.nz)

    k2facx = jnp.concatenate((
        -(jnp.arange(0, (grids.nx//2)) * kfacx) ** 2,
        -(jnp.arange((grids.nx//2), 0, -1) * kfacx) ** 2
    ))[:,jnp.newaxis,jnp.newaxis]

    k2facy = jnp.concatenate((
        -(jnp.arange(0, (grids.ny//2)) * kfacy) ** 2,
        -(jnp.arange((grids.ny//2), 0, -1) * kfacy) ** 2
    ))[jnp.newaxis,:,jnp.newaxis]

    k2facz = jnp.concatenate((
        -(jnp.arange(0, (grids.nz//2)) * kfacz) ** 2,
        -(jnp.arange((grids.nz//2), 0, -1) * kfacz) ** 2
    ))[jnp.newaxis,jnp.newaxis,:]

    if grids.bangx > 0.000001 or grids.bangy > 0.000001 or grids.bangz > 0.000001:
        pass
    else:
        psout = jnp.copy(psin)

    psout = psout.at[...].set(
        jnp.fft.fftn(psout, axes=(-3, -2, -1))
    )

    psout = psout.at[...].divide(
        ((weight * (e0inv - forces.h2ma * (k2facx + k2facy + k2facz))) + 0.5 * weightuv * v_pairmax) *
        (grids.nx * grids.ny * grids.nz)
    )

    psout = psout.at[...].set(
        jnp.fft.ifftn(psout, axes=(-3, -2, -1), norm='forward')
    )

    if grids.bangx > 0.000001 or grids.bangy > 0.000001 or grids.bangz > 0.000001:
        pass

    return psout

def cdervx00(d: float, psin: jax.Array) -> jax.Array:
    """Calculate x derivative using FFT. Works with both 3D and 4D arrays."""
    shape = psin.shape
    if len(shape) == 3:  # Handle 3D input (nx, ny, nz)
        nx, ny, nz = shape
        psin = psin.reshape(1, nx, ny, nz)  # Add dummy iq dimension
    elif len(shape) == 4:  # Handle 4D input (iq, nx, ny, nz)
        iq, nx, ny, nz = shape
    else:
        raise ValueError(f"Input array must be 3D or 4D, got shape {shape}")
        
    kfac = (jnp.pi + jnp.pi) / (d * nx)
    half_n = nx // 2

    d1psout = jnp.fft.fft(psin, axis=1, norm="backward")

    inds = jnp.arange(0, half_n)
    inds4d = inds[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d1psout = d1psout.at[:,0,:,:].set(0.0)
    d1psout = d1psout.at[:,inds,:,:].multiply((1j * inds4d) * kfac / nx)
    d1psout = d1psout.at[:,nx-inds,:,:].multiply(-((1j * inds4d) * kfac) / nx)

    d1psout = d1psout.at[:,half_n,:,:].set(0.0)
    d1psout = jnp.fft.ifft(d1psout, axis=1, norm="forward")
    
    if len(shape) == 3:  # Remove dummy dimension if input was 3D
        d1psout = d1psout[0]
        
    return d1psout

def cdervy00(d: float, psin: jax.Array) -> jax.Array:
    """Calculate y derivative using FFT. Works with both 3D and 4D arrays."""
    shape = psin.shape
    if len(shape) == 3:  # Handle 3D input (nx, ny, nz)
        nx, ny, nz = shape
        psin = psin.reshape(1, nx, ny, nz)  # Add dummy iq dimension
    elif len(shape) == 4:  # Handle 4D input (iq, nx, ny, nz)
        iq, nx, ny, nz = shape
    else:
        raise ValueError(f"Input array must be 3D or 4D, got shape {shape}")
        
    kfac = (jnp.pi + jnp.pi) / (d * ny)
    half_n = ny // 2

    d1psout = jnp.fft.fft(psin, axis=2, norm="backward")

    inds = jnp.arange(0, half_n)
    inds4d = inds[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]

    d1psout = d1psout.at[:,:,0,:].set(0.0)
    d1psout = d1psout.at[:,:,inds,:].multiply((1j * inds4d) * kfac / ny)
    d1psout = d1psout.at[:,:,ny-inds,:].multiply(-((1j * inds4d) * kfac) / ny)

    d1psout = d1psout.at[:,:,half_n,:].set(0.0)
    d1psout = jnp.fft.ifft(d1psout, axis=2, norm="forward")
    
    if len(shape) == 3:  # Remove dummy dimension if input was 3D
        d1psout = d1psout[0]
        
    return d1psout

def cdervz00(d: float, psin: jax.Array) -> jax.Array:
    """Calculate z derivative using FFT. Works with both 3D and 4D arrays."""
    shape = psin.shape
    if len(shape) == 3:  # Handle 3D input (nx, ny, nz)
        nx, ny, nz = shape
        psin = psin.reshape(1, nx, ny, nz)  # Add dummy iq dimension
    elif len(shape) == 4:  # Handle 4D input (iq, nx, ny, nz)
        iq, nx, ny, nz = shape
    else:
        raise ValueError(f"Input array must be 3D or 4D, got shape {shape}")
        
    kfac = (jnp.pi + jnp.pi) / (d * nz)
    half_n = nz // 2

    d1psout = jnp.fft.fft(psin, axis=3, norm="backward")

    inds = jnp.arange(0, half_n)
    inds4d = inds[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d1psout = d1psout.at[:,:,:,0].set(0.0)
    d1psout = d1psout.at[:,:,:,inds].multiply((1j * inds4d) * kfac / nz)
    d1psout = d1psout.at[:,:,:,nz-inds].multiply(-((1j * inds4d) * kfac) / nz)

    d1psout = d1psout.at[:,:,:,half_n].set(0.0)
    d1psout = jnp.fft.ifft(d1psout, axis=3, norm="forward")
    
    if len(shape) == 3:  # Remove dummy dimension if input was 3D
        d1psout = d1psout[0]
        
    return d1psout

@partial(jax.jit, static_argnums=(1,2,3,4,5,6))
def calculate_derivatives(psin: jax.Array, nx: int, ny: int, nz: int, 
                        dx: float, dy: float, dz: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculate first derivatives in all directions using FFT method.
    Works with both 3D and 4D input arrays.
    
    Parameters:
    psin: Input wavefunction (3D or 4D array)
    nx, ny, nz: Number of grid points in each direction
    dx, dy, dz: Grid spacings
    """
    d1x = cdervx00(dx, psin)
    d1y = cdervy00(dy, psin) 
    d1z = cdervz00(dz, psin)
    
    return d1x, d1y, d1z

@partial(jax.jit, static_argnums=(1,2,3,4,5,6))
def calculate_orbital_momentum(levels: Levels, nx: int, ny: int, nz: int,
                             dx: float, dy: float, dz: float,
                             x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Calculate orbital angular momentum components for all states."""
    
    def orbital_components(psi):
        # Calculate derivatives
        d1x, d1y, d1z = calculate_derivatives(psi, nx, ny, nz, dx, dy, dz)
        
        # Reshape coordinates for broadcasting
        x_3d = x[:,jnp.newaxis,jnp.newaxis]
        y_3d = y[jnp.newaxis,:,jnp.newaxis]
        z_3d = z[jnp.newaxis,jnp.newaxis,:]
        
        # Calculate orbital angular momentum components
        lx = jnp.sum(jnp.real(jnp.conj(psi) * (y_3d * d1z - z_3d * d1y)))
        ly = jnp.sum(jnp.real(jnp.conj(psi) * (z_3d * d1x - x_3d * d1z)))
        lz = jnp.sum(jnp.real(jnp.conj(psi) * (x_3d * d1y - y_3d * d1x)))
        
        return jnp.array([lx, ly, lz])
    
    return jax.vmap(orbital_components)(levels.psi)


@partial(jax.jit, static_argnames=['grids'])
def calculate_position_varying_derivatives(grids, psin: jax.Array, pos_func: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculate position-varying derivatives in all directions.
    Returns d2x, d2y, d2z derivatives.
    """
    _, d2x = cdervx02(grids.dx, psin, pos_func)
    _, d2y = cdervy02(grids.dy, psin, pos_func)
    _, d2z = cdervz02(grids.dz, psin, pos_func)
    
    return d2x, d2y, d2z


@jax.jit
def laplace_fft(grids, forces, wg: float, wguv: float, psin: jax.Array, 
                v_pairmax: float, e0inv: Optional[float] = None) -> jax.Array:
    """
    Calculate Laplacian or inverse operator using FFT method.
    If e0inv is not provided, calculates Laplacian.
    If e0inv is provided, calculates inverse operator.
    """
    if grids.bangx > 0.000001 or grids.bangy > 0.000001 or grids.bangz > 0.000001:
        raise ValueError('Laplace does not work with Bloch boundaries.')

    weightmin = 0.1
    weight = jnp.maximum(wg, weightmin)
    weightuv = jnp.maximum(wguv, weightmin)

    # Calculate k-space factors
    kfacx = (jnp.pi + jnp.pi) / (grids.dx * grids.nx)
    kfacy = (jnp.pi + jnp.pi) / (grids.dy * grids.ny)
    kfacz = (jnp.pi + jnp.pi) / (grids.dz * grids.nz)

    # Calculate k^2 factors for each direction
    k2facx = jnp.concatenate([
        -(jnp.arange(grids.nx//2) * kfacx) ** 2,
        -(jnp.arange(grids.nx//2, 0, -1) * kfacx) ** 2
    ])
    k2facy = jnp.concatenate([
        -(jnp.arange(grids.ny//2) * kfacy) ** 2,
        -(jnp.arange(grids.ny//2, 0, -1) * kfacy) ** 2
    ])
    k2facz = jnp.concatenate([
        -(jnp.arange(grids.nz//2) * kfacz) ** 2,
        -(jnp.arange(grids.nz//2, 0, -1) * kfacz) ** 2
    ])

    # Reshape for broadcasting
    k2facx = k2facx[:, jnp.newaxis, jnp.newaxis]
    k2facy = k2facy[jnp.newaxis, :, jnp.newaxis]
    k2facz = k2facz[jnp.newaxis, jnp.newaxis, :]

    # Transform to k-space
    psout = jnp.fft.fftn(psin, axes=(-3, -2, -1))

    # Apply operator
    if e0inv is not None:
        denominator = (weight * (e0inv - forces.h2ma * (k2facx + k2facy + k2facz)) + 
                      0.5 * weightuv * v_pairmax)
        psout = psout / (denominator * (grids.nx * grids.ny * grids.nz))
    else:
        psout = psout * (k2facx + k2facy + k2facz) / (grids.nx * grids.ny * grids.nz)

    # Transform back to real space
    psout = jnp.fft.ifftn(psout, axes=(-3, -2, -1))

    return psout
