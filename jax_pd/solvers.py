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

class State(namedtuple('State', ['position', 'velocity', 'acceleration', 'mass'])):
    """A tuple containing the state of an simulation.
    Attributes:
    position: An ndarray of shape [n, spatial_dimension] storing the position
        of particles.
    velocity: An ndarray of shape [n, spatial_dimension] storing the velocity
        of particles.
    acceleration: An ndarray of shape [n, spatial_dimension] storing the
        acceleration of particles from the previous step.
    mass: A float or an ndarray of shape [n] containing the masses of the
        particles.
    """

    def __new__(cls, position, velocity, acceleration, mass):
        return super(State, cls).__new__(cls, position, velocity, acceleration, mass)

register_pytree_namedtuple(State)


def apply_bc_fun(state, t=f32(0), Rbc=[], Vbc=[]):
    R, V, A, mass = state
    R = onp.array(R)
    V = onp.array(V)
    for i in Rbc:
        R[i['bool'],:] = i['fn'](R[i['bool'],:],i['bool'],t=t,**i['kwargs'])
    for i in Vbc:
        V[i['bool'],:] = i['fn'](V[i['bool'],:],i['bool'],t=t,**i['kwargs'])
    return State(R, V, A, mass)


# pylint: disable=invalid-name
def velocity_verlet(energy_or_force, shift_fn, dt, quant=quantity.Energy):
    """Simulates a system using velocity_verlet.
    Args:
        energy_or_force: A function that produces either an energy or a force from
            a set of particle positions specified as an ndarray of shape
            [n, spatial_dimension].
        shift_fn: A function that displaces positions, R, by an amount dR. Both R
            and dR should be ndarrays of shape [n, spatial_dimension].
        dt: Floating point number specifying the timescale (step size) of the
            simulation.
        quant: Either a quantity.Energy or a quantity.Force specifying whether
            energy_or_force is an energy or force respectively.
    Returns:
        See above.
    """
    force = [quantity.canonicalize_force(i, quant) for i in energy_or_force]


    dt, = static_cast(dt)
    dt_2, = static_cast(0.5 * dt)

    def init_fun(key, Ri, velocity=f32(0.0), mass=f32(1.0), **kwargs):
        Vi = velocity + np.zeros(Ri.shape, dtype=Ri.dtype)
        mass = quantity.canonicalize_mass(mass)
        Ai = sum([i(Ri, t=0, **kwargs) for i in force]) / mass
        return State(Ri, Vi, Ai, mass)

    def update_RA(state, t=f32(0.0), **kwargs):
        Ri, Vi, Ai, mass = state
        Vi2 = Vi + Ai * dt_2
        Rj = shift_fn(Ri, Vi2 * dt, t=t, **kwargs)
        Aj = sum([i(Rj, t=t, **kwargs) for i in force]) / mass
        return State(Rj, Vi2, Aj, mass)

    def update_V(state, t=f32(0.0), **kwargs):
        Rj, Vi2, Aj, mass = state
        Vj = Vi2 + Aj * dt_2
        return State(Rj, Vj, Aj, mass)

    return init_fun, update_RA, update_V


def run_simulation(key,runs,time_step,R,energy_fn,shift_fn,Vbc=[],Rbc=[],iVbc=[],iRbc=[],mass=1.0,print_every=1,value_every=1,pos_every=1,solver=None):
    bonds, stretch = energy_fn[-1]
    energy_fn_calables = [i['func'](*i['args'],**i['kwargs']) for i in energy_fn[:-1]]

    if solver==None:
        init, inc_RA, inc_V = velocity_verlet(energy_fn_calables, shift_fn, time_step)
    else:
        init, inc_RA, inc_V = solver(energy_fn_calables, shift_fn, time_step)

    inc_RA_c = jit(inc_RA)
    inc_V_c = jit(inc_V)

    R1 = []
    S1 = []
    state_values = []
    state = init(key, R, mass=mass)
    print('Initial conditions(after iBC): ')
    state = apply_bc_fun(state,0,Vbc=iVbc,Rbc=iRbc)
    print(pd_util.describe_state(state,energy_fn_calables))
    state = apply_bc_fun(state,0,Vbc=Vbc,Rbc=Rbc)
    print('Initial conditions(after BC): ')
    print(pd_util.describe_state(state,energy_fn_calables))
    S1 = [state]

    state_values += [pd_util.get_state_values(state,energy_fn_calables)]

    for i in range(runs):
        state = inc_RA_c(state,t=i)
        state = apply_bc_fun(state,i,Rbc=Rbc)
        state = inc_V_c(state,t=i)
        state = apply_bc_fun(state,i,Vbc=Vbc)

        if i % 5==0:
            lengths = pd_util.bond_lengths(bonds,state.position)
            for ind in range(len(energy_fn[:-1])):
                if 'length' in energy_fn[ind]['kwargs'].keys():
                    mask = lengths < energy_fn[ind]['kwargs']['length']*(stretch+1.0)
                    bonds = bonds[mask]
                    energy_fn[ind]['kwargs']['length'] = energy_fn[ind]['kwargs']['length'][mask]
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

    return R1,state_values,S1
