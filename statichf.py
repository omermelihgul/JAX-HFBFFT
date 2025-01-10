from densities import add_density_jit
from static import diagstep_jit

def statichf(coulomb, densities, energies, forces, grids, levels, meanfield, params, static):
    if params.trestart:
        firstiter = params.iteration + 1
    else:
        params.iteration = 0
        firstiter = 1

    energies, levels, static = diagstep_jit(energies, forces, grids, levels, static, False, True)

    densities = add_density_jit(densities, grids, levels)

    skyrme()


    sp_properties



    pass
