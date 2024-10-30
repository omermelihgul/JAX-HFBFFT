import jax
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, statichf

jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_platform_name', 'cpu')

config = read_yaml('_config.yml')

# initialization
params = init_params(**config.get('params', {}))
forces = init_forces(params, **config.get('force', {}))
grids = init_grids(params, **config.get('grids', {}))
densities = init_densities(grids)
meanfield = init_meanfield(grids)
levels = init_levels(grids, **config.get('levels', {}))
static = init_static(levels, **config.get('static', {}))

# determine wave function numbers etc
if params.nof > params.mnof:
    raise ValueError('nof > mnof.')
elif params.nof > 0:
    pass
else:
    pass

# initialize wave functions
if params.nof > 0:
    pass
elif params.nof == 0:
    print('1283')
elif params.nof == -1:
    pass
else:
    pass

# coulomb initialization
if params.tcoul:
    pass

# static or dynamic  calculation performed
statichf(params)

from test import *
from levels import cdervz02


pinn = load4d('pinn')
meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
meanfield.aq = meanfield.aq.at[...].set(load5d_real('aq'))
meanfield.v_pair = meanfield.v_pair.at[...].set(load4d_real('v_pair'))
pout = load4d('pout')
pout_mf = load4d('pout_mf')
#pout_del = load4d('pout_del')

#pswk = load4d('pswk')
#pswk2 = load4d('pswk2')

from meanfield import hpsi00_jit

res, res1 = hpsi00_jit(grids, meanfield, 0, 1.0, 0.0, pinn)

def relative_norm_error(res, pout):
    norm_diff = jnp.linalg.norm(res - pout)
    norm_pout = jnp.linalg.norm(pout)
    if norm_pout == 0:
        return jnp.inf
    relative_error = norm_diff / norm_pout
    return relative_error

error = relative_norm_error(res, pout)
print("Relative Norm Error:", error)

error = relative_norm_error(res1, pout_mf)
print("Relative Norm Error:", error)

#error = relative_norm_error(res2, pout_del)
#print("Relative Norm Error:", error)

print(jnp.max(jnp.abs(pout - res)))
print(jnp.max(jnp.abs(pout_mf - res1)))
#print(jnp.max(jnp.abs(pout_del - res2)))

#print(jnp.max(res1))
#print(jnp.max(pswk))
#print(jnp.max(res2))
#print(jnp.max(pswk2))









'''
pinn = load4d('pinn')
pswk = load4d('pswk')
pswk2 = load4d('pswk2')
bmass = load4d_real('bmass')

res, res2 = cdervz02(grids.dz, pinn, bmass[0,...])
print(jnp.max(jnp.abs(res - pswk)))
print(jnp.max(jnp.abs(res2 - pswk2)))
'''


































# print('\n')
# print(params)
# print('\n')
# print(grids.der1x[0,:])
# print('\n')
# print(levels.isospin)



#test
#from levels import cdervz00_jit
#from test import *

#pinn = load4d('pinn')
#pswk2 = load4d('pswk2')
#pswk = load4d('pswk')
#pos_func = load3d('pos_func')

#res = cdervz00_jit(0.8, pswk2)
#omer
#print(jnp.max(jnp.abs(pswk - res)))
#print(jnp.max(pswk))
#print(jnp.max(res))
#print(jnp.max(jnp.abs(pswk2 - res2)))
#print(jnp.max(pswk2))
#print(jnp.max(res2))

#levels.psi = levels.psi.at[...].set(load5d('psi'))
#meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
#meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
#meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
#meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
#meanfield.aq = meanfield.aq.at[...].set(load5d_real('aq'))
#pout = load4d('pout')


#from meanfield import hpsi0fft, hpsi0fft_jit


#res = hpsi0fft(grids, meanfield, 0, levels.psi[0,...])

#print(jnp.max(jnp.abs(pout - res)))


#from levels import laplace


#print(laplace(grids))
#print(grids.bangx)



#ps1 = load4d('ps1')
#ps2 = load4d('ps2')

#res = laplace(grids, forces, 1.0 , 0.0, ps1, 0, 100.0)


#fft_result = jnp.fft.ifftn(psout, axes=(-3, -2, -1), norm='forward')
#print(jnp.max(jnp.abs(res - ps2)))


#difference = res[0,:,1,11] - ps2[0,:,1,11]
#norm_difference = np.linalg.norm(difference, ord=2)
#norm_exact = np.linalg.norm(ps2[0,:,1,11], ord=2)

#relative_error = norm_difference / norm_exact
#print(f"L2 Relative Norm Error: {relative_error:.4f}")

#relative_error = jnp.abs(res[0,...] - ps2[0,...]) / jnp.abs(res[0,...])
#max_relative_error = jnp.nanmax(relative_error)
#print("Maximum Relative Error:", max_relative_error)

#print(res[0,11,:,14])
#print(fft_result[0,11,:,14])
