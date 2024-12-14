import jax
import jax.numpy as jnp
from test import *
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, statichf
from coulomb import init_coulomb
from moment import Moment
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
moment = Moment(jnp.array([-1.4219041817327197E-016, -5.6374715900987012E-017, 3.9732369799358789E-016]))

def cdervx01(d, psin):
    _, n, _, _ = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=1, norm="backward")

    idx = jnp.arange(0, half_n)
    idx_4d = idx[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,idx,:,:].multiply(-(idx_4d * kfac) ** 2 / n)
    d2psout = d2psout.at[:,n-1-idx,:,:].multiply(-((idx_4d + 1) * kfac) ** 2 / n)

    d1psout = d1psout.at[:,0,:,:].set(0.0)
    d1psout = d1psout.at[:,idx,:,:].multiply((1j * idx_4d) * kfac / n)
    d1psout = d1psout.at[:,n-idx,:,:].multiply(-((1j * idx_4d) * kfac) / n)

    d1psout = d1psout.at[:,half_n,:,:].set(0.0)

    d1psout = d1psout.at[...].set(
        jnp.fft.ifft(d1psout, axis=1, norm="forward")
    )

    d2psout = d2psout.at[...].set(
        jnp.fft.ifft(d2psout, axis=1, norm="forward")
    )

    # 5.2245375141434353e-17
    # 1.36640756370672e-16
    return d1psout, d2psout

def cdervy01(d, psin):
    _, _, n, _ = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=2, norm="backward")

    idx = jnp.arange(0, half_n)
    idx_4d = idx[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,idx,:].multiply(-(idx_4d * kfac) ** 2 / n)
    d2psout = d2psout.at[:,:,n-1-idx,:].multiply(-((idx_4d + 1) * kfac) ** 2 / n)

    d1psout = d1psout.at[:,:,0,:].set(0.0)
    d1psout = d1psout.at[:,:,idx,:].multiply((1j * idx_4d) * kfac / n)
    d1psout = d1psout.at[:,:,n-idx,:].multiply(-((1j * idx_4d) * kfac) / n)

    d1psout = d1psout.at[:,:,half_n,:].set(0.0)

    d1psout = d1psout.at[...].set(
        jnp.fft.ifft(d1psout, axis=2, norm="forward")
    )

    d2psout = d2psout.at[...].set(
        jnp.fft.ifft(d2psout, axis=2, norm="forward")
    )

    # 3.966240216616929e-17
    # 1.170194769447655e-16
    return d1psout, d2psout

def cdervz01(d, psin):
    _, _, _, n = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=3, norm="backward")

    idx = jnp.arange(0, half_n)
    idx_4d = idx[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,:,idx].multiply(-(idx_4d * kfac) ** 2 / n)
    d2psout = d2psout.at[:,:,:,n-1-idx].multiply(-((idx_4d + 1) * kfac) ** 2 / n)

    d1psout = d1psout.at[:,:,:,0].set(0.0)
    d1psout = d1psout.at[:,:,:,idx].multiply((1j * idx_4d) * kfac / n)
    d1psout = d1psout.at[:,:,:,n-idx].multiply(-((1j * idx_4d) * kfac) / n)

    d1psout = d1psout.at[:,:,:,half_n].set(0.0)

    d1psout = d1psout.at[...].set(
        jnp.fft.ifft(d1psout, axis=3, norm="forward")
    )

    d2psout = d2psout.at[...].set(
        jnp.fft.ifft(d2psout, axis=3, norm="forward")
    )

    # 4.105394205386439e-17
    # 1.258475307421716e-16
    return d1psout, d2psout




def sp_properties_helper(pst, dx, dy, dz, xx, yy, zz):
    psx, psw = cdervx01(dx, pst)
    psy, ps2 = cdervy01(dy, pst)
    psw = psw.at[...].add(ps2)
    psz, ps3 = cdervz01(dz, pst)
    psw = psw.at[...].add(ps3)

    cc = jnp.array([
        jnp.sum(
            jnp.real(pst) * (yy * jnp.imag(psz) - zz * jnp.imag(psy)) +
            jnp.imag(pst) * (zz * jnp.real(psy) - yy * jnp.real(psz))
        ),
        jnp.sum(
            jnp.real(pst) * (zz * jnp.imag(psx) - xx * jnp.imag(psz)) +
            jnp.imag(pst) * (xx * jnp.real(psz) - zz * jnp.real(psx))
        ),
        jnp.sum(
            jnp.real(pst) * (xx * jnp.imag(psy) - yy * jnp.imag(psx)) +
            jnp.imag(pst) * (yy * jnp.real(psx) - xx * jnp.real(psy))
        )
    ])

    kin = -jnp.sum(
        jnp.real(pst) * jnp.real(psw) + jnp.imag(pst) * jnp.imag(psw)
    )

    xpar = jnp.sum(
        jnp.real(pst) * jnp.real(pst[:, ::-1, ::-1, ::-1]) +
        jnp.imag(pst) * jnp.imag(pst[:, ::-1, ::-1, ::-1])
    )

    ss = jnp.array([
        jnp.sum(
            jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...]) +
            jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...])
        ),
        jnp.sum(
            jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...] * (0.0 - 1.0j)) +
            jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...] * (0.0 + 1.0j))
        ),
        jnp.sum(
            jnp.real(jnp.conjugate(pst[0,...]) * pst[0,...]) -
            jnp.real(jnp.conjugate(pst[1,...]) * pst[1,...])
        )
    ])

    return ss, cc, kin, xpar

sp_properties_helper_vmap = jax.vmap(
    sp_properties_helper,
    in_axes=(0, None, None, None, None, None, None)
)

@jax.jit
def sp_properties(forces, grids, levels, moment):
    xx = (grids.x - moment.cmtot[0])[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
    yy = (grids.y - moment.cmtot[1])[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]
    zz = (grids.z - moment.cmtot[2])[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]


    ss, cc, kin, xpar = sp_properties_helper_vmap(
        levels.psi, grids.dx, grids.dy, grids.dz, xx, yy, zz
    )

    levels.sp_spin = levels.sp_spin.at[...].set(
        0.5 * grids.wxyz * ss
    )
    levels.sp_orbital = levels.sp_orbital.at[...].set(
        grids.wxyz * cc
    )
    levels.sp_kinetic = levels.sp_kinetic.at[...].set(
        grids.wxyz * jnp.where(levels.isospin == 0, forces.h2m[0], forces.h2m[1]) * kin
    )
    levels.sp_parity = levels.sp_parity.at[...].set(
        grids.wxyz * xpar
    )

    return levels



levels.psi = levels.psi.at[...].set(load5d('psi'))

sp_spin = load2d_real('sp_spin', 3, 132)
sp_orbital = load2d_real('sp_orbital', 3, 132)
sp_kinetic = load1d_real('sp_kinetic', 132)
sp_parity = load1d_real('sp_parity', 132)

res = sp_properties(forces, grids, levels, moment)

print("sp_spin")
print(jnp.max(jnp.abs(res.sp_spin - sp_spin)))

print("sp_orbital")
print(jnp.max(jnp.abs(res.sp_orbital - sp_orbital)))

print("sp_kinetic")
print(jnp.max(jnp.abs(res.sp_kinetic - sp_kinetic)))

print("sp_parity")
print(jnp.max(jnp.abs(res.sp_parity - sp_parity)))


#relative_error = jnp.abs(res.sp_spin.flatten() - sp_spin.flatten()) / jnp.abs(sp_spin.flatten())

#print(jnp.max(relative_error))

print(sp_spin[5,:])
print(res.sp_spin[5,:])
'''


pst = load4d('pst')
psx = load4d('psx')
psy = load4d('psy')
psz = load4d('psz')
psw = load4d('psw')
ps2 = load4d('ps2')

xx = load1d_real('xx')[jnp.newaxis, :, np.newaxis, np.newaxis]
yy = load1d_real('yy')[jnp.newaxis, np.newaxis, :, np.newaxis]
zz = load1d_real('zz')[jnp.newaxis, np.newaxis, np.newaxis, :]

sp_spin = load1d_real('sp_spin', 3)
sp_orbital = load1d_real('sp_orbital', 3)
sp_kinetic = load1d_real('sp_kinetic', 1)
sp_parity = load1d_real('sp_parity', 1)


wxyz = 0.8 * 0.8 * 0.8

kin = 0.0
xpar = 0.0

cc = jnp.array([
    jnp.sum(
        jnp.real(pst) * (yy * jnp.imag(psz) - zz * jnp.imag(psy)) +
        jnp.imag(pst) * (zz * jnp.real(psy) - yy * jnp.real(psz))
    ),
    jnp.sum(
        jnp.real(pst) * (zz * jnp.imag(psx) - xx * jnp.imag(psz)) +
        jnp.imag(pst) * (xx * jnp.real(psz) - zz * jnp.real(psx))
    ),
    jnp.sum(
        jnp.real(pst) * (xx * jnp.imag(psy) - yy * jnp.imag(psx)) +
        jnp.imag(pst) * (yy * jnp.real(psx) - xx * jnp.real(psy))
    )
])

kin = - jnp.sum(
    jnp.real(pst) * jnp.real(psw) + jnp.imag(pst) * jnp.imag(psw)
)

xpar = jnp.sum(
    jnp.real(pst) * jnp.real(pst[:, ::-1, ::-1, ::-1]) +
    jnp.imag(pst) * jnp.imag(pst[:, ::-1, ::-1, ::-1])
)

ss = jnp.array([
    jnp.sum(
        jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...]) +
        jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...])
    ),
    jnp.sum(
        jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...] * (0.0 - 1.0j)) +
        jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...] * (0.0 + 1.0j))
    ),
    jnp.sum(
        jnp.real(jnp.conjugate(pst[0,...]) * pst[0,...]) -
        jnp.real(jnp.conjugate(pst[1,...]) * pst[1,...])
    )
])


print(kin)
print(xpar)
print(cc)
print(sp_spin)
print(sp_orbital)
print(sp_kinetic)
print(sp_parity)


'''
