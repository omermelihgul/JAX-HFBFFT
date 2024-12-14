import jax
import jax.numpy as jnp
from test import *
jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_platform_name', 'cpu')



pst = load4d('pst')
psx = load4d('psx')
psy = load4d('psy')
psz = load4d('psz')
psw = load4d('psw')
ps2 = load4d('ps2')

xx = load1d_real('xx')
yy = load1d_real('yy')
zz = load1d_real('zz')

sp_spin = load1d_real('sp_spin', 3)
sp_orbital = load1d_real('sp_orbital', 3)
sp_kinetic = load1d_real('sp_kinetic', 1)
sp_parity = load1d_real('sp_parity', 1)


wxyz = 0.8 * 0.8 * 0.8

ss = jnp.zeros(3, dtype=jnp.float64)
cc = jnp.zeros(3, dtype=jnp.float64)
kin = 0.0
xpar = 0.0

cc = jnp.array([
    jnp.sum(
        pst *
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
print(ss)
print(sp_spin)
print(sp_orbital)
print(sp_kinetic)
print(sp_parity)
