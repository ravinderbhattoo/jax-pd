
"""Definitions of various standard energy functions for material decription."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import jax.numpy as np

from jax_md import space, smap
from jax_md.interpolate import spline
from jax_md.util import *


def simple_spring(
    dr, length=f32(1), stretch=f32(0.5), epsilon=f32(1), alpha=f32(2), radius=f32(1), epsilon2=f32(1), alpha2=f32(2), **unused_kwargs):
    """Isotropic spring potential with a given rest length and maximum stretch.
    We define `simple_spring` to be a generalized Hookian spring with
    agreement when alpha = 2.
    """
    check_kwargs_time_dependence(unused_kwargs)
    dr0 = dr / radius
    U1 = epsilon2 * np.where((dr0 < 1.0), f32(1.0) / alpha2 * (f32(1.0) - dr0) ** alpha2, f32(0.0))
    U2 = np.where((dr - length)/length<stretch, epsilon / alpha * (dr - length) ** alpha,f32(0.0))
    return U2 - U1

def simple_spring_bond(displacement_or_metric, bond, bond_type=None, length=f32(1), stretch=f32(0.5), epsilon=f32(1.0), alpha=f32(2.0), radius=f32(1.0), epsilon2=f32(1.0), alpha2=f32(2.0)):
    """Convenience wrapper to compute energy of particles bonded by springs."""
    length = np.array(length, f32)
    epsilon = np.array(epsilon, f32)
    alpha = np.array(alpha, f32)
    radius = np.array(radius, f32)
    epsilon2 = np.array(epsilon2, f32)
    alpha2 = np.array(alpha2, f32)
    return smap.bond(simple_spring,
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        bond,
        bond_type,
        length=length,
        stretch=stretch,
        epsilon=epsilon,
        alpha=alpha,
        epsilon2=epsilon2,
        alpha2=alpha2,
        radius=radius)

def anti_contact_simple_spring(dr, radius=f32(1.0), epsilon=f32(1.0), alpha=f32(2.0), **unused_kwargs):
    """Finite ranged repulsive interaction between soft spheres.
    Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Particle radii. Should either be a floating point scalar or an
        ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
        or an ndarray whose shape is [n, m].
    alpha: Exponent specifying interaction stiffness. Should either be a float
        point scalar or an ndarray whose shape is [n, m].
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
    Returns:
    Matrix of energies whose shape is [n, m].
    """
    check_kwargs_time_dependence(unused_kwargs)
    dr = dr / radius
    U = epsilon * np.where((dr < 1.0), - f32(1.0) / alpha * (f32(1.0) - dr) ** alpha, f32(0.0))
    return U


def anti_contact_simple_spring_bond(displacement_or_metric, bond, bond_type=None, radius=f32(1.0), epsilon=f32(1.0), alpha=f32(2.0)):
    """Convenience wrapper to compute energy of particles bonded by springs."""
    epsilon = np.array(epsilon, f32)
    radius = np.array(radius, f32)
    alpha = np.array(alpha, f32)
    return smap.bond(anti_contact_simple_spring,
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        bond,
        bond_type,
        radius=radius,
        epsilon=epsilon,
        alpha=alpha)
