import jax
from math import pi
from functools import partial
from dataclasses import dataclass
from jax.tree_util import register_dataclass


@partial(jax.tree_util.register_dataclass,
         data_fields=[],
         meta_fields=['pi',
                      'hbc',
                      'e2',
                      'r0',
                      'wffile',
                      'converfile',
                      'monopolesfile',
                      'dipolesfile',
                      'momentafile',
                      'energiesfile',
                      'diffenergiesfile',
                      'quadrupolesfile',
                      'spinfile',
                      'extfieldfile',
                      'tabcfile',
                      'scratch',
                      'scratch2',
                      'tcoul',
                      'tstatic',
                      'tdynamic',
                      'tfft',
                      'trestart',
                      'mprint',
                      'mplot',
                      'mrest',
                      'iteration',
                      'time',
                      'wflag',
                      'printnow',
                      'nselect',
                      'writeselect',
                      'write_isospin',
                      'mnof',
                      'nof',
                      'tabc_nprocs',
                      'tabc_myid'])
@dataclass
class Params:
    # useful constants
    pi: float
    hbc: float
    e2: float
    r0: float

    # names of files and units to be used
    wffile: str
    converfile: str
    monopolesfile: str
    dipolesfile: str
    momentafile: str
    energiesfile: str
    diffenergiesfile: str
    quadrupolesfile: str
    spinfile: str
    extfieldfile: str
    tabcfile: str

    scratch: int
    scratch2: int

    # basic parameters controlling the job
    tcoul: bool
    tstatic: bool
    tdynamic: bool
    tfft: bool
    trestart: bool

    # parameters controlling printout frequency etc
    mprint: int
    mplot: int
    mrest: int
    iteration: int
    time: float
    wflag: bool
    printnow: bool
    nselect: int
    writeselect: str
    write_isospin: bool
    mnof: int
    nof: int
    tabc_nprocs: int
    tabc_myid: int

def init_params(imode=1, **kwargs) -> Params:
    default_kwargs = {
        'pi': pi,
        'hbc': 197.328910,
        'e2': 1.4399784085965135,
        'r0': 1.2,
        'wffile': 'none',
        'converfile': 'conver',
        'monopolesfile': 'monopoles',
        'dipolesfile': 'dipoles',
        'momentafile': 'momenta',
        'energiesfile': 'energies',
        'diffenergiesfile': 'diffenergies',
        'quadrupolesfile': 'quadrupoles',
        'spinfile': 'spin',
        'extfieldfile': 'extfield',
        'tabcfile': 'tabc',
        'scratch': 11,
        'scratch2': 12,
        'tcoul': True,
        'tfft': True,
        'trestart': False,
        'mprint': 100,
        'mplot': 0,
        'mrest': 0,
        'iteration': 0,
        'time': None,
        'wflag': True,
        'printnow': False,
        'nselect': 10,
        'writeselect': 'r',
        'write_isospin': False,
        'mnof': 4,
        'nof': 0,
        'tabc_nprocs': 1,
        'tabc_myid': 0
    }

    default_kwargs.update(kwargs)

    default_kwargs['tstatic'] = True if imode == 1 else False
    default_kwargs['tdynamic'] = True if imode == 0 else False

    return Params(**default_kwargs)
