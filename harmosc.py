import jax
from jax import numpy as jnp
from itertools import product
from test import *

nshell = load2d_int('nshell', 3, 132)

# print(nshell)


def generate_nshell(npsi: jax.Array) -> jax.Array:
    nst = 0
    nshell = []

    for iq in range(2):
        nps = npsi[iq]
        exit = False

        for ka in range(nps + 1):
            for i, j, k in product(range(ka + 1), repeat=3):
                if i + j + k == ka:
                    for _ in range(2):
                        nst += 1
                        if nst > nps:
                            exit = True
                            break
                        nshell.append([k, j, i])
                    if exit:
                        break
            if exit:
                break
        nst -= 1

    return jnp.array(nshell)



res = generate_nshell(jnp.array([82,132]))

print((nshell == res).all())

print(nshell[:5,:])
print(res[:5,:])

# print(nshell.shape)
# def harmosc(npsi: jax.Array):

#     nst = 0
#     nshell = []
#     for iq in range(2):
#         nps = npsi[iq]

#         for ka in range(nps + 1):



def harmosc(grids, static):
    pass
