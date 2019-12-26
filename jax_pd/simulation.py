from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from jax import jit
from jax import ops
from jax import random
import jax.numpy as np

from jax_md import quantity
from jax_md import interpolate

from jax_md.util import *

from jax_pd import util as pd_util
from jax_pd.solvers import *


def run(info, key,runs,time_step,R,energy_fn,shift_fn,
            Vbc=[],Rbc=[],iVbc=[],iRbc=[],mass=1.0,
            print_every=1,value_every=1,pos_every=1,solver=None):

    R1, S1, state_values = info

    bonds, stretch = energy_fn[-1]
    energy_fn_calables = [i['func'](*i['args'],**i['kwargs']) for i in energy_fn[:-1]]

    if solver==None:
        init, inc_RA, inc_V = velocity_verlet(energy_fn_calables, shift_fn, time_step)
    else:
        init, inc_RA, inc_V = solver(energy_fn_calables, shift_fn, time_step)

    inc_RA_c = jit(inc_RA)
    inc_V_c = jit(inc_V)

    state = init(key, R, mass=mass)
    print('Initial conditions(after iBC): ')
    state = apply_bc_fun(state,0,Vbc=iVbc,Rbc=iRbc)
    print(pd_util.describe_state(state,energy_fn_calables))
    state = apply_bc_fun(state,0,Vbc=Vbc,Rbc=Rbc)
    print('Initial conditions(after BC): ')
    print(pd_util.describe_state(state,energy_fn_calables))

    R1 += [state.position]
    S1 += [state]
    state_values += [pd_util.get_state_values(state,energy_fn_calables)]

    for i in range(runs):
        state = inc_RA_c(state,t=i)
        state = apply_bc_fun(state,i*time_step,Rbc=Rbc)
        state = inc_V_c(state,t=i)
        state = apply_bc_fun(state,i*time_step,Vbc=Vbc)

        if i % 10 == 0:
            lengths = pd_util.bond_lengths(bonds,state.position)
            for ind in range(len(energy_fn[:-1])):
                if 'length' in energy_fn[ind]['kwargs'].keys():
                    intact_bonds = lengths < energy_fn[ind]['kwargs']['length']*(stretch+1.0)
                    bonds = bonds[intact_bonds]
                    energy_fn[ind]['kwargs']['length'] = energy_fn[ind]['kwargs']['length'][intact_bonds]
                    energy_fn[ind]['kwargs']['bond'] = bonds
            energy_fn_calables = [j['func'](*j['args'],**j['kwargs']) for j in energy_fn[:-1]]

            if solver==None:
                init, inc_RA, inc_V = velocity_verlet(energy_fn_calables, shift_fn, time_step)
            else:
                init, inc_RA, inc_V = solver(energy_fn_calables, shift_fn, time_step)
            inc_RA_c = jit(inc_RA)
            inc_V_c = jit(inc_V)

        if i % value_every == 0 or i==(runs-1):
            state_values += [pd_util.get_state_values(state,energy_fn_calables)]
        if i % pos_every == 0 or i==(runs-1):
            R1 += [state.position]
            S1 += [state]
        if i % print_every == 0 or i==(runs-1):
            print('Run #',i,': ',pd_util.describe_state(state,energy_fn_calables))

    return True
