def sp_properties(
    grids,
    levels,
    moment,
):
    levels.sp_orbital = levels.sp_orbital.at[...].set(0.0)
    levels.sp_spin = levels.sp_spin.at[...].set(0.0)
    levels.sp_kinetic = levels.sp_kinetic.at[...].set(0.0)
    levels.sp_parity = levels.sp_parity.at[...].set(0.0)

    xx = grids.x - moment.cmtot[0]
    yy = grids.y - moment.cmtot[1]
    zz = grids.z - moment.cmtot[2]

    pass
