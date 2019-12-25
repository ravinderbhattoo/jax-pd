from functools import reduce, partial
from jax import lax, ops, vmap, eval_shape
from jax_md.util import *
from jax_md import quantity, partition
import numpy as onp

def distance(a,b):
    return np.sqrt(((a-b)**2).sum())

def r_d(info,ind):
    return info>ind

v_r_d = vmap(partial(r_d),(0,0))

def bond_lengths(bonds,R):
    _metric = vmap(partial(distance), 0, 0)
    R_ = onp.array(R)
    Ra = R_[bonds[:, 0]]
    Rb = R_[bonds[:, 1]]
    return _metric(Ra,Rb)

def bonds(displacement_fn, box_size, cutoff, R):
    neigh = partition.neighbor_list(displacement_fn,box_size,cutoff,R)
    info = neigh(R)
    info_ = np.zeros(info.shape,i32) + np.arange(0,len(info)).reshape(-1,1)
    mask =  (info<len(R)) & (v_r_d(info,np.arange(len(R))))
    bonds = onp.array(np.vstack([info_[mask],info[mask]]).T)

    return bonds


def describe_state(state,energy_fn):
    PE = sum([i(state.position) for i in energy_fn])
    KE = quantity.kinetic_energy(state.velocity,state.mass)
    return 'KE: {:3f}, PE: {:.3f}, TE: {:.3f}'.format(KE,PE,KE+PE)


def get_state_values(state,energy_fn):
    PE = sum([i(state.position) for i in energy_fn])
    KE = quantity.kinetic_energy(state.velocity,state.mass)
    return KE,PE,KE+PE


def write_trj(R1):
    with open('./datafile.xyz','w+') as f:
        for r in R1:
            f.write(str(len(r))+'\n\n')
            for i in r:
                f.write('{} {} {}\n'.format(*i))

#
